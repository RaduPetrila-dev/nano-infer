// nano-infer: the GPT-2 forward pass.
#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/weights.hpp"

namespace nanoinfer {

// Called after each stage the reference dump can check, with the work already
// synchronised. Each call costs a stream synchronise, so this is a debugging
// hook and not something a decode loop should carry.
//
//   stage   "embed.out", "h.<i>.out", "ln_f.out" or "logits"
//   device  device pointer to rows * cols contiguous floats
using StageHook = std::function<void(const std::string& stage,
                                     const float* device, int rows, int cols)>;

// One prefill pass over a prompt.
//
// Every activation lives in a single device allocation sized at construction,
// so a forward allocates nothing. `weights` must outlive the model.
//
// There is no KV cache yet, so each call starts at position zero and recomputes
// the whole prompt. Generation arrives with the cache.
class GPT2Model {
 public:
  // max_tokens fixes the workspace. The logits buffer dominates it at
  // max_tokens * n_vocab floats, so ask for the prompt length and no more.
  //
  // Throws std::runtime_error on a non-f32 checkpoint or a max_tokens outside
  // 1..n_ctx.
  GPT2Model(const ModelWeights& weights, int max_tokens);

  GPT2Model(const GPT2Model&) = delete;
  GPT2Model& operator=(const GPT2Model&) = delete;

  // ids are host-side, `tokens` of them, each below n_vocab. Returns once the
  // logits are readable on the device.
  void forward(const int* ids, int tokens, const StageHook& hook = {});

  // [tokens, n_vocab] on the device, valid until the next forward.
  const float* logits() const noexcept { return logits_; }

  int tokens() const noexcept { return tokens_; }
  int max_tokens() const noexcept { return max_tokens_; }
  std::size_t workspace_bytes() const noexcept { return arena_.used(); }

 private:
  void run_block(const LayerWeights& layer, int tokens);
  void emit(const StageHook& hook, const std::string& stage, const float* data,
            int rows, int cols);

  const ModelWeights& weights_;
  int max_tokens_ = 0;
  int tokens_ = 0;

  DeviceArena arena_;
  CudaStream stream_;
  PinnedBuffer<int> ids_host_;

  int* ids_ = nullptr;       // [max_tokens]
  float* x_ = nullptr;       // [max_tokens, d_model], the residual stream
  float* norm_ = nullptr;    // [max_tokens, d_model]
  float* attn_ = nullptr;    // [max_tokens, d_model], merged head output
  float* branch_ = nullptr;  // [max_tokens, d_model], sub-layer output
  float* qkv_ = nullptr;     // [max_tokens, 3 * d_model]
  float* mlp_ = nullptr;     // [max_tokens, 4 * d_model]
  float* scores_ = nullptr;  // [n_head, max_tokens, max_tokens]
  float* logits_ = nullptr;  // [max_tokens, n_vocab]
};

}  // namespace nanoinfer
