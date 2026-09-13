// nano-infer: unfused attention launcher.
#pragma once

#include <cuda_runtime.h>

namespace nanoinfer {

// Geometry of one attention call.
//
// Q, K and V carry separate base pointers and row strides. The packed
// [Q | K | V] row the QKV projection writes and a standalone KV cache are then
// the same call with different strides, not two code paths. Head h owns columns
// [h * head_dim, (h + 1) * head_dim) of every row.
//
// Strides are in elements and must each cover n_head * head_dim. kv_stride
// applies to K and V alike, since they always arrive from the same buffer.
//
// Query row r sits at absolute position pos_offset + r and sees keys 0 through
// pos_offset + r. Prefill passes pos_offset 0 with n_key == n_query. Decode
// passes one query row with pos_offset set to the number of cached tokens.
// Omitting the offset makes every decode step attend to key 0 alone.
struct AttentionShape {
  int n_query = 0;
  int n_key = 0;
  int n_head = 0;
  int head_dim = 0;
  int q_stride = 0;
  int kv_stride = 0;
  int out_stride = 0;
  int pos_offset = 0;

  int model_dim() const { return n_head * head_dim; }

  // Scratch the scores buffer needs, laid out [head][query][key]. The same
  // order HuggingFace reports attention probabilities in.
  long long score_elements() const {
    return static_cast<long long>(n_head) * n_query * n_key;
  }

  // Prefill over rows of [Q | K | V], each n_head * head_dim wide.
  static AttentionShape packed(int tokens, int n_head, int head_dim);
};

// Throws std::runtime_error when the shape cannot describe a legal call. Every
// launcher below runs it first, so a bad shape costs no launch.
void attention_validate(const AttentionShape& shape);

// out[r][h] = softmax(Q[r][h] · K[·][h]ᵀ / sqrt(head_dim)) * V[·][h]
//
//   out     [n_query, out_stride]  must not alias q, k, v or scores
//   scores  scratch, shape.score_elements() floats, holds the softmax
//           probabilities on return
//   q       [n_query, q_stride]
//   k, v    [n_key, kv_stride]
//
// Launch is asynchronous. Synchronise the stream before reading `out` on the
// host.
void attention_forward(float* out, float* scores, const float* q, const float* k,
                       const float* v, const AttentionShape& shape,
                       cudaStream_t stream);

// The same call against one packed [Q | K | V] buffer, which is what
// gemm_forward writes for the QKV projection.
void attention_forward_packed(float* out, float* scores, const float* qkv,
                              int tokens, int n_head, int head_dim,
                              cudaStream_t stream);

// The stages, exposed so a failing test names the one at fault. attention_scores
// writes -inf into masked positions and attention_softmax replaces those with
// exact zeros.
void attention_scores(float* scores, const float* q, const float* k,
                      const AttentionShape& shape, cudaStream_t stream);

void attention_softmax(float* scores, const AttentionShape& shape,
                       cudaStream_t stream);

void attention_context(float* out, const float* probs, const float* v,
                       const AttentionShape& shape, cudaStream_t stream);

}  // namespace nanoinfer
