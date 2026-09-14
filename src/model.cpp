// The forward pass. No device code here, only launches, so this is a .cpp.
//
// Block order follows GPT2Block.forward: two residual branches, each one
// LayerNorm, a sub-layer and an add back into the stream. Getting the order
// wrong produces a model that runs, converges on nothing and matches no
// reference, so docs/forward.md writes it out next to the HuggingFace source.

#include "nanoinfer/model.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "nanoinfer/kernels/attention.cuh"
#include "nanoinfer/kernels/embedding.cuh"
#include "nanoinfer/kernels/gelu.cuh"
#include "nanoinfer/kernels/gemm.cuh"
#include "nanoinfer/kernels/layernorm.cuh"
#include "nanoinfer/kernels/residual.cuh"

namespace nanoinfer {
namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("nano-infer model: " + message);
}

// Arena allocations are 256-byte aligned, so rounding here makes the total
// exact rather than approximate.
std::size_t arena_bytes(long long count, std::size_t element) {
  return align_up(static_cast<std::size_t>(count) * element, 256);
}

}  // namespace

GPT2Model::GPT2Model(const ModelWeights& weights, int max_tokens)
    : weights_(weights), max_tokens_(max_tokens) {
  const GPT2Config& cfg = weights.config();

  if (weights.dtype() != DType::F32) {
    fail(std::string("checkpoint is ") + dtype_name(weights.dtype()) +
         ", the kernels are f32 only");
  }
  if (cfg.n_layer == 0 || cfg.n_head == 0 || cfg.n_embd == 0 ||
      cfg.n_vocab == 0 || cfg.n_ctx == 0) {
    fail("checkpoint declares a zero dimension");
  }
  if (weights.tensors().layers.size() != cfg.n_layer) {
    fail("checkpoint resolved " + std::to_string(weights.tensors().layers.size()) +
         " layers, header says " + std::to_string(cfg.n_layer));
  }
  if (max_tokens <= 0 ||
      static_cast<std::uint32_t>(max_tokens) > cfg.n_ctx) {
    fail("max_tokens " + std::to_string(max_tokens) + " is outside 1.." +
         std::to_string(cfg.n_ctx));
  }

  const long long rows = max_tokens;
  const long long d = cfg.n_embd;

  const std::size_t id_bytes = arena_bytes(rows, sizeof(int));
  const std::size_t stream_bytes = arena_bytes(rows * d, sizeof(float));
  const std::size_t qkv_bytes = arena_bytes(rows * 3 * d, sizeof(float));
  const std::size_t mlp_bytes = arena_bytes(rows * cfg.ffn_dim(), sizeof(float));
  const std::size_t score_bytes =
      arena_bytes(static_cast<long long>(cfg.n_head) * rows * rows, sizeof(float));
  const std::size_t logit_bytes = arena_bytes(rows * cfg.n_vocab, sizeof(float));

  // x, norm, attn and branch are each one stream row wide.
  arena_.reset(id_bytes + 4 * stream_bytes + qkv_bytes + mlp_bytes + score_bytes +
               logit_bytes);

  ids_ = reinterpret_cast<int*>(arena_.alloc(id_bytes));
  x_ = reinterpret_cast<float*>(arena_.alloc(stream_bytes));
  norm_ = reinterpret_cast<float*>(arena_.alloc(stream_bytes));
  attn_ = reinterpret_cast<float*>(arena_.alloc(stream_bytes));
  branch_ = reinterpret_cast<float*>(arena_.alloc(stream_bytes));
  qkv_ = reinterpret_cast<float*>(arena_.alloc(qkv_bytes));
  mlp_ = reinterpret_cast<float*>(arena_.alloc(mlp_bytes));
  scores_ = reinterpret_cast<float*>(arena_.alloc(score_bytes));
  logits_ = reinterpret_cast<float*>(arena_.alloc(logit_bytes));

  ids_host_.allocate(static_cast<std::size_t>(max_tokens));
}

void GPT2Model::emit(const StageHook& hook, const std::string& stage,
                     const float* data, int rows, int cols) {
  if (!hook) return;
  stream_.sync();
  hook(stage, data, rows, cols);
}

void GPT2Model::run_block(const LayerWeights& layer, int tokens) {
  const GPT2Config& cfg = weights_.config();
  const int d = static_cast<int>(cfg.n_embd);
  const int ff = static_cast<int>(cfg.ffn_dim());
  const int heads = static_cast<int>(cfg.n_head);
  const int head_dim = static_cast<int>(cfg.head_dim());
  cudaStream_t stream = stream_.get();

  // Attention branch. The stream itself is never normalised in place, since the
  // residual add at the end needs the values that went in.
  layernorm_forward(norm_, x_, layer.ln1_w.as<float>(), layer.ln1_b.as<float>(),
                    tokens, d, kLayerNormEps, stream);
  gemm_forward(qkv_, norm_, layer.qkv_w.as<float>(), layer.qkv_b.as<float>(),
               tokens, 3 * d, d, stream);
  attention_forward_packed(attn_, scores_, qkv_, tokens, heads, head_dim, stream);
  gemm_forward(branch_, attn_, layer.attn_proj_w.as<float>(),
               layer.attn_proj_b.as<float>(), tokens, d, d, stream);
  residual_add(x_, x_, branch_, tokens * d, stream);

  // MLP branch.
  layernorm_forward(norm_, x_, layer.ln2_w.as<float>(), layer.ln2_b.as<float>(),
                    tokens, d, kLayerNormEps, stream);
  gemm_forward(mlp_, norm_, layer.fc_w.as<float>(), layer.fc_b.as<float>(),
               tokens, ff, d, stream);
  gelu_forward(mlp_, mlp_, tokens * ff, stream);
  gemm_forward(branch_, mlp_, layer.proj_w.as<float>(),
               layer.proj_b.as<float>(), tokens, d, ff, stream);
  residual_add(x_, x_, branch_, tokens * d, stream);
}

void GPT2Model::forward(const int* ids, int tokens, const StageHook& hook) {
  if (ids == nullptr) fail("ids is null");
  if (tokens <= 0 || tokens > max_tokens_) {
    fail("tokens " + std::to_string(tokens) + " is outside 1.." +
         std::to_string(max_tokens_));
  }
  tokens_ = tokens;

  const GPT2Config& cfg = weights_.config();
  const GPT2Weights& w = weights_.tensors();
  const int d = static_cast<int>(cfg.n_embd);
  const int vocab = static_cast<int>(cfg.n_vocab);
  cudaStream_t stream = stream_.get();

  // Staged through pinned memory so the copy runs on the stream rather than
  // blocking the caller. Out-of-range ids are caught by a device assert inside
  // the embedding kernel, which release builds drop.
  std::copy(ids, ids + tokens, ids_host_.data());
  CUDA_CHECK(cudaMemcpyAsync(ids_, ids_host_.data(),
                             static_cast<std::size_t>(tokens) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  embedding_forward(x_, ids_, w.wte.as<float>(), w.wpe.as<float>(), tokens, d,
                    /*pos_offset=*/0, vocab, static_cast<int>(cfg.n_ctx), stream);
  emit(hook, "embed.out", x_, tokens, d);

  for (std::size_t i = 0; i < w.layers.size(); ++i) {
    run_block(w.layers[i], tokens);
    emit(hook, "h." + std::to_string(i) + ".out", x_, tokens, d);
  }

  layernorm_forward(norm_, x_, w.ln_f_w.as<float>(), w.ln_f_b.as<float>(), tokens,
                    d, kLayerNormEps, stream);
  emit(hook, "ln_f.out", norm_, tokens, d);

  // lm_head ties to wte, so the head reads the embedding table back as its
  // weight and contracts along a stored row. No bias on this one.
  gemm_forward_bt(logits_, norm_, w.wte.as<float>(), nullptr, tokens, vocab, d,
                  stream);

  stream_.sync();
  emit(hook, "logits", logits_, tokens, vocab);
}

}  // namespace nanoinfer
