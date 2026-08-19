#include "nanoinfer/weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace nanoinfer {
namespace {

// 32 MiB per staging buffer. Large enough to saturate PCIe, small enough that
// two of them stay cheap to pin.
constexpr std::size_t kStageChunk = 32ull << 20;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("nano-infer checkpoint: " + message);
}

std::string layer_key(std::uint32_t layer, const char* suffix) {
  return "h." + std::to_string(layer) + "." + suffix;
}

}  // namespace

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

MappedFile::MappedFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    fail("cannot open " + path + ": " + std::strerror(errno));
  }

  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    fd_ = -1;
    fail("cannot stat " + path + ": " + std::strerror(errno));
  }
  size_ = static_cast<std::size_t>(st.st_size);

  void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapping == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    size_ = 0;
    fail("cannot map " + path + ": " + std::strerror(errno));
  }
  data_ = static_cast<std::byte*>(mapping);

  // The whole file gets read once, front to back. Tell the kernel so it can
  // read ahead aggressively instead of faulting in page by page.
  ::madvise(data_, size_, MADV_SEQUENTIAL | MADV_WILLNEED);
}

MappedFile::~MappedFile() { release(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), data_(other.data_), size_(other.size_) {
  other.fd_ = -1;
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    release();
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MappedFile::release() noexcept {
  if (data_ != nullptr) {
    ::munmap(data_, size_);
    data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  size_ = 0;
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

const TensorView* ModelWeights::find(const std::string& name) const {
  auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &it->second;
}

ModelWeights ModelWeights::load(const std::string& path) {
  ModelWeights model;
  model.file_ = MappedFile(path);

  const std::byte* base = model.file_.data();
  const std::size_t file_size = model.file_.size();

  if (file_size < format::kHeaderBytes) {
    fail("file is shorter than the header");
  }

  format::Header header{};
  std::memcpy(&header, base, sizeof(header));

  if (header.magic != format::kMagic) {
    fail("bad magic, this is not a nano-infer checkpoint");
  }
  if (header.version != format::kVersion) {
    fail("version " + std::to_string(header.version) + " is unsupported, expected " +
         std::to_string(format::kVersion));
  }
  if (header.dtype > static_cast<std::uint32_t>(DType::BF16)) {
    fail("unknown dtype tag " + std::to_string(header.dtype));
  }
  if (header.n_head == 0 || header.n_embd % header.n_head != 0) {
    fail("n_embd must divide evenly by n_head");
  }

  model.dtype_ = static_cast<DType>(header.dtype);
  model.config_ = GPT2Config{header.n_layer, header.n_head, header.n_embd,
                             header.n_ctx, header.n_vocab};

  const std::size_t dir_bytes =
      static_cast<std::size_t>(header.n_tensors) * format::kEntryBytes;
  const std::size_t dir_end = format::kHeaderBytes + dir_bytes;
  if (dir_end > file_size) {
    fail("tensor directory runs past the end of the file");
  }

  const std::size_t data_start = align_up(dir_end, format::kAlign);
  if (data_start > file_size) {
    fail("data region starts past the end of the file");
  }
  const std::size_t data_bytes = file_size - data_start;

  // One allocation for every weight. Offsets inside the checkpoint are already
  // 256-byte aligned, so the same relative offsets hold on the device.
  model.arena_.reset(data_bytes);
  std::byte* arena_base = model.arena_.alloc(data_bytes, format::kAlign);

  // Read the directory before touching the GPU. A malformed entry then costs
  // nothing.
  model.by_name_.reserve(header.n_tensors * 2);
  for (std::uint32_t i = 0; i < header.n_tensors; ++i) {
    format::Entry entry{};
    std::memcpy(&entry, base + format::kHeaderBytes + i * format::kEntryBytes,
                sizeof(entry));

    // A name that fills the field has no terminator, so bound the search.
    const std::size_t name_len =
        ::strnlen(entry.name, format::kNameBytes);
    std::string name(entry.name, name_len);
    if (name.empty()) {
      fail("tensor " + std::to_string(i) + " has an empty name");
    }
    if (entry.offset < data_start || entry.offset + entry.nbytes > file_size) {
      fail("tensor '" + name + "' points outside the data region");
    }
    if (entry.ndim == 0 || entry.ndim > format::kMaxDims) {
      fail("tensor '" + name + "' has rank " + std::to_string(entry.ndim));
    }

    TensorView view;
    view.host = base + entry.offset;
    view.device = arena_base + (entry.offset - data_start);
    view.nbytes = entry.nbytes;
    view.ndim = entry.ndim;
    std::copy(std::begin(entry.dims), std::end(entry.dims), std::begin(view.dims));

    const std::uint64_t expected = view.numel() * dtype_size(model.dtype_);
    if (expected != entry.nbytes) {
      fail("tensor '" + name + "' declares " + std::to_string(entry.nbytes) +
           " bytes but its shape implies " + std::to_string(expected));
    }

    if (!model.by_name_.emplace(std::move(name), view).second) {
      fail("duplicate tensor name in the directory");
    }
  }

  // Double-buffered upload. While one chunk moves over PCIe, the next is being
  // copied out of the page cache into the other pinned buffer.
  if (data_bytes > 0) {
    const std::size_t chunk = std::min(kStageChunk, data_bytes);
    PinnedBuffer<std::byte> stage[2] = {PinnedBuffer<std::byte>(chunk),
                                        PinnedBuffer<std::byte>(chunk)};
    CudaEvent copied[2] = {CudaEvent(), CudaEvent()};
    CudaStream stream;

    std::size_t offset = 0;
    int slot = 0;
    while (offset < data_bytes) {
      const std::size_t n = std::min(chunk, data_bytes - offset);
      copied[slot].sync();  // the previous transfer out of this buffer is done
      std::memcpy(stage[slot].data(), base + data_start + offset, n);
      CUDA_CHECK(cudaMemcpyAsync(arena_base + offset, stage[slot].data(), n,
                                 cudaMemcpyHostToDevice, stream.get()));
      copied[slot].record(stream.get());
      offset += n;
      slot ^= 1;
    }
    stream.sync();
  }

  // Resolve the flat name table into the structure the kernels consume.
  auto take = [&](const std::string& name, std::initializer_list<std::uint32_t> shape) {
    const TensorView* view = model.find(name);
    if (view == nullptr) {
      fail("missing tensor '" + name + "'");
    }
    if (view->ndim != shape.size()) {
      fail("tensor '" + name + "' has rank " + std::to_string(view->ndim) +
           ", expected " + std::to_string(shape.size()));
    }
    std::uint32_t axis = 0;
    for (std::uint32_t want : shape) {
      if (view->dims[axis] != want) {
        fail("tensor '" + name + "' axis " + std::to_string(axis) + " is " +
             std::to_string(view->dims[axis]) + ", expected " +
             std::to_string(want));
      }
      ++axis;
    }
    return *view;
  };

  const GPT2Config& cfg = model.config_;
  const std::uint32_t d = cfg.n_embd;
  const std::uint32_t ff = cfg.ffn_dim();

  GPT2Weights& w = model.tensors_;
  w.wte = take("wte", {cfg.n_vocab, d});
  w.wpe = take("wpe", {cfg.n_ctx, d});
  w.ln_f_w = take("ln_f.w", {d});
  w.ln_f_b = take("ln_f.b", {d});

  w.layers.resize(cfg.n_layer);
  for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
    LayerWeights& layer = w.layers[i];
    layer.ln1_w = take(layer_key(i, "ln_1.w"), {d});
    layer.ln1_b = take(layer_key(i, "ln_1.b"), {d});
    layer.qkv_w = take(layer_key(i, "attn.qkv.w"), {d, 3 * d});
    layer.qkv_b = take(layer_key(i, "attn.qkv.b"), {3 * d});
    layer.attn_proj_w = take(layer_key(i, "attn.proj.w"), {d, d});
    layer.attn_proj_b = take(layer_key(i, "attn.proj.b"), {d});
    layer.ln2_w = take(layer_key(i, "ln_2.w"), {d});
    layer.ln2_b = take(layer_key(i, "ln_2.b"), {d});
    layer.fc_w = take(layer_key(i, "mlp.fc.w"), {d, ff});
    layer.fc_b = take(layer_key(i, "mlp.fc.b"), {ff});
    layer.proj_w = take(layer_key(i, "mlp.proj.w"), {ff, d});
    layer.proj_b = take(layer_key(i, "mlp.proj.b"), {d});
  }

  return model;
}

}  // namespace nanoinfer
