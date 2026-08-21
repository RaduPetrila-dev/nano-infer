// nano-infer: model configuration and checkpoint format constants.
//
// tools/export_gpt2.py writes the format described here. 

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace nanoinfer {

enum class DType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
};

inline std::size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::F32:
      return 4;
    case DType::F16:
    case DType::BF16:
      return 2;
  }
  return 0;
}

inline const char* dtype_name(DType dt) {
  switch (dt) {
    case DType::F32:
      return "f32";
    case DType::F16:
      return "f16";
    case DType::BF16:
      return "bf16";
  }
  return "unknown";
}

// HuggingFace GPT2Config.layer_norm_epsilon. Not a free parameter: change it
// and the logits stop matching the reference.
constexpr float kLayerNormEps = 1e-5f;

struct GPT2Config {
  std::uint32_t n_layer = 0;
  std::uint32_t n_head = 0;
  std::uint32_t n_embd = 0;
  std::uint32_t n_ctx = 0;
  std::uint32_t n_vocab = 0;

  std::uint32_t head_dim() const { return n_embd / n_head; }
  std::uint32_t ffn_dim() const { return 4 * n_embd; }

  // Bytes of KV cache for one sequence at full context length.
  std::size_t kv_cache_bytes(DType dt) const {
    return static_cast<std::size_t>(n_layer) * 2 * n_ctx * n_embd *
           dtype_size(dt);
  }
};

// ---------------------------------------------------------------------------
// Checkpoint format v1
// ---------------------------------------------------------------------------
//
//   [0, 64)                     header
//   [64, 64 + 96 * n_tensors)   tensor directory
//   align to 256                data region, each tensor 256-byte aligned
//
// All integers are little-endian. Tensor offsets are absolute file offsets.

namespace format {

constexpr std::uint32_t kMagic = 0x494E414Eu;  // "NANI"
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderBytes = 64;
constexpr std::size_t kEntryBytes = 96;
constexpr std::size_t kNameBytes = 48;
constexpr std::size_t kAlign = 256;
constexpr std::size_t kMaxDims = 4;

#pragma pack(push, 1)
struct Header {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t dtype;
  std::uint32_t n_layer;
  std::uint32_t n_head;
  std::uint32_t n_embd;
  std::uint32_t n_ctx;
  std::uint32_t n_vocab;
  std::uint32_t n_tensors;
  std::uint32_t reserved[7];
};

struct Entry {
  char name[kNameBytes];
  std::uint64_t offset;
  std::uint64_t nbytes;
  std::uint32_t ndim;
  std::uint32_t dims[kMaxDims];
  std::uint32_t reserved[3];
};
#pragma pack(pop)

static_assert(sizeof(Header) == kHeaderBytes, "header size drifted from spec");
static_assert(sizeof(Entry) == kEntryBytes, "entry size drifted from spec");

}  // namespace format

}  // namespace nanoinfer
