// nano-infer: checkpoint loading.
//
// The checkpoint is mapped, uploaded to one contiguous device allocation, then
// carved into named views. Host pointers stay valid for as long as the
// ModelWeights object lives, which keeps CPU-side verification cheap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"

namespace nanoinfer {

// ---------------------------------------------------------------------------
// Read-only memory mapping (POSIX)
// ---------------------------------------------------------------------------

class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  const std::byte* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

 private:
  void release() noexcept;

  int fd_ = -1;
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Tensor views
// ---------------------------------------------------------------------------

struct TensorView {
  const std::byte* host = nullptr;  // into the mapped checkpoint
  const std::byte* device = nullptr;  // into the weight arena
  std::uint64_t nbytes = 0;
  std::uint32_t ndim = 0;
  std::uint32_t dims[format::kMaxDims] = {0, 0, 0, 0};

  std::uint64_t numel() const {
    std::uint64_t n = ndim == 0 ? 0 : 1;
    for (std::uint32_t i = 0; i < ndim; ++i) n *= dims[i];
    return n;
  }

  template <typename T>
  const T* as() const {
    return reinterpret_cast<const T*>(device);
  }

  template <typename T>
  const T* host_as() const {
    return reinterpret_cast<const T*>(host);
  }
};

// One transformer block. Naming follows the maths, not the HuggingFace module
// tree, so the kernel code reads cleanly.
struct LayerWeights {
  TensorView ln1_w, ln1_b;
  TensorView qkv_w, qkv_b;    // [n_embd, 3 * n_embd], [3 * n_embd]
  TensorView attn_proj_w, attn_proj_b;  // [n_embd, n_embd], [n_embd]
  TensorView ln2_w, ln2_b;
  TensorView fc_w, fc_b;      // [n_embd, 4 * n_embd], [4 * n_embd]
  TensorView proj_w, proj_b;  // [4 * n_embd, n_embd], [n_embd]
};

struct GPT2Weights {
  TensorView wte;  // [n_vocab, n_embd], also the tied output projection
  TensorView wpe;  // [n_ctx, n_embd]
  TensorView ln_f_w, ln_f_b;
  std::vector<LayerWeights> layers;
};

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

class ModelWeights {
 public:
  ModelWeights() = default;
  ModelWeights(ModelWeights&&) = default;
  ModelWeights& operator=(ModelWeights&&) = default;
  ModelWeights(const ModelWeights&) = delete;
  ModelWeights& operator=(const ModelWeights&) = delete;

  // Throws std::runtime_error on a malformed checkpoint, CudaError on a failed
  // allocation or transfer.
  static ModelWeights load(const std::string& path);

  const GPT2Config& config() const noexcept { return config_; }
  DType dtype() const noexcept { return dtype_; }
  const GPT2Weights& tensors() const noexcept { return tensors_; }
  std::size_t device_bytes() const noexcept { return arena_.used(); }
  std::size_t tensor_count() const noexcept { return by_name_.size(); }

  // Returns nullptr when the name is absent.
  const TensorView* find(const std::string& name) const;

 private:
  MappedFile file_;
  DeviceArena arena_;
  GPT2Config config_{};
  DType dtype_ = DType::F32;
  GPT2Weights tensors_{};
  std::unordered_map<std::string, TensorView> by_name_;
};

}  // namespace nanoinfer
