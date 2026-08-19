// Round-trip test: build a tiny synthetic checkpoint on disk, load it, then
// check that every tensor arrived on the device with the right bytes.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/weights.hpp"

using namespace nanoinfer;

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", what);
    ++g_failures;
  }
}

struct PendingTensor {
  std::string name;
  std::vector<std::uint32_t> dims;
  std::vector<float> data;
};

std::uint64_t elements(const std::vector<std::uint32_t>& dims) {
  std::uint64_t n = 1;
  for (std::uint32_t d : dims) n *= d;
  return n;
}

PendingTensor make_tensor(const std::string& name,
                          std::vector<std::uint32_t> dims, float seed) {
  PendingTensor t;
  t.name = name;
  t.dims = std::move(dims);
  const std::uint64_t n = elements(t.dims);
  t.data.resize(n);
  for (std::uint64_t i = 0; i < n; ++i) {
    t.data[i] = seed + static_cast<float>(i % 97) * 0.25f;
  }
  return t;
}

void write_checkpoint(const std::string& path, const GPT2Config& cfg,
                      const std::vector<PendingTensor>& tensors) {
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    throw std::runtime_error("cannot write " + path);
  }

  format::Header header{};
  header.magic = format::kMagic;
  header.version = format::kVersion;
  header.dtype = static_cast<std::uint32_t>(DType::F32);
  header.n_layer = cfg.n_layer;
  header.n_head = cfg.n_head;
  header.n_embd = cfg.n_embd;
  header.n_ctx = cfg.n_ctx;
  header.n_vocab = cfg.n_vocab;
  header.n_tensors = static_cast<std::uint32_t>(tensors.size());

  const std::size_t dir_end =
      format::kHeaderBytes + tensors.size() * format::kEntryBytes;
  std::size_t cursor = align_up(dir_end, format::kAlign);

  std::vector<format::Entry> entries(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const PendingTensor& t = tensors[i];
    format::Entry& e = entries[i];
    std::memset(&e, 0, sizeof(e));
    std::strncpy(e.name, t.name.c_str(), format::kNameBytes);
    e.offset = cursor;
    e.nbytes = t.data.size() * sizeof(float);
    e.ndim = static_cast<std::uint32_t>(t.dims.size());
    for (std::size_t d = 0; d < t.dims.size(); ++d) e.dims[d] = t.dims[d];
    cursor = align_up(cursor + e.nbytes, format::kAlign);
  }

  std::fwrite(&header, sizeof(header), 1, out);
  std::fwrite(entries.data(), sizeof(format::Entry), entries.size(), out);

  const std::vector<char> pad(format::kAlign, 0);
  std::size_t written = dir_end;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    while (written < entries[i].offset) {
      const std::size_t n = std::min(pad.size(), entries[i].offset - written);
      std::fwrite(pad.data(), 1, n, out);
      written += n;
    }
    std::fwrite(tensors[i].data.data(), 1, entries[i].nbytes, out);
    written += entries[i].nbytes;
  }

  std::fclose(out);
}

void compare_device(const TensorView& view, const std::vector<float>& expected,
                    const char* label) {
  std::vector<float> actual(expected.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), view.device, view.nbytes,
                        cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL %s: element %zu is %f, expected %f\n", label, i,
                   static_cast<double>(actual[i]),
                   static_cast<double>(expected[i]));
      ++g_failures;
      return;
    }
  }
}

}  // namespace

int main() {
  const GPT2Config cfg{/*n_layer=*/2, /*n_head=*/2, /*n_embd=*/8, /*n_ctx=*/16,
                       /*n_vocab=*/32};
  const std::uint32_t d = cfg.n_embd;
  const std::uint32_t ff = cfg.ffn_dim();

  std::vector<PendingTensor> tensors;
  tensors.push_back(make_tensor("wte", {cfg.n_vocab, d}, 1.0f));
  tensors.push_back(make_tensor("wpe", {cfg.n_ctx, d}, 2.0f));
  tensors.push_back(make_tensor("ln_f.w", {d}, 3.0f));
  tensors.push_back(make_tensor("ln_f.b", {d}, 4.0f));

  for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
    const std::string p = "h." + std::to_string(i) + ".";
    const float s = static_cast<float>(i + 1) * 10.0f;
    tensors.push_back(make_tensor(p + "ln_1.w", {d}, s + 1.0f));
    tensors.push_back(make_tensor(p + "ln_1.b", {d}, s + 2.0f));
    tensors.push_back(make_tensor(p + "attn.qkv.w", {d, 3 * d}, s + 3.0f));
    tensors.push_back(make_tensor(p + "attn.qkv.b", {3 * d}, s + 4.0f));
    tensors.push_back(make_tensor(p + "attn.proj.w", {d, d}, s + 5.0f));
    tensors.push_back(make_tensor(p + "attn.proj.b", {d}, s + 6.0f));
    tensors.push_back(make_tensor(p + "ln_2.w", {d}, s + 7.0f));
    tensors.push_back(make_tensor(p + "ln_2.b", {d}, s + 8.0f));
    tensors.push_back(make_tensor(p + "mlp.fc.w", {d, ff}, s + 9.0f));
    tensors.push_back(make_tensor(p + "mlp.fc.b", {ff}, s + 10.0f));
    tensors.push_back(make_tensor(p + "mlp.proj.w", {ff, d}, s + 11.0f));
    tensors.push_back(make_tensor(p + "mlp.proj.b", {d}, s + 12.0f));
  }

  const std::string path = "nano_infer_roundtrip.bin";

  try {
    write_checkpoint(path, cfg, tensors);
    ModelWeights model = ModelWeights::load(path);

    check(model.config().n_layer == cfg.n_layer, "layer count survived");
    check(model.config().n_embd == cfg.n_embd, "d_model survived");
    check(model.tensor_count() == tensors.size(), "tensor count survived");
    check(model.tensors().layers.size() == cfg.n_layer, "layers resolved");

    for (const PendingTensor& t : tensors) {
      const TensorView* view = model.find(t.name);
      if (view == nullptr) {
        std::fprintf(stderr, "FAIL missing tensor %s\n", t.name.c_str());
        ++g_failures;
        continue;
      }
      compare_device(*view, t.data, t.name.c_str());
    }
  } catch (const std::exception& err) {
    std::fprintf(stderr, "FAIL exception: %s\n", err.what());
    ++g_failures;
  }

  std::remove(path.c_str());

  if (g_failures == 0) {
    std::printf("PASS test_weights\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", g_failures);
  return 1;
}
