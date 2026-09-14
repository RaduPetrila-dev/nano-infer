// nano-infer: GEMM launcher.
#pragma once

#include <cuda_runtime.h>

namespace nanoinfer {

// C[m, n] = A[m, k] * B[k, n] + bias[n]
//
// Row-major throughout, no transpose on either operand. HuggingFace GPT-2 stores
// every projection weight as Conv1D, which is [in_features, out_features] and
// computes y = x @ W, so the exporter writes it untransposed and this kernel
// reads it as B directly. See the layout convention in README.md.
//
//   c     [m, n]  must not alias a, b or bias
//   a     [m, k]  activations, m = tokens, k = input features
//   b     [k, n]  weight, k = input features, n = output features
//   bias  [n]     optional, pass nullptr to skip it
//
// The four shapes a 124M forward pass uses, per block:
//
//   attn.qkv   [t, 768]  * [768, 2304]
//   attn.proj  [t, 768]  * [768, 768]
//   mlp.fc     [t, 768]  * [768, 3072]
//   mlp.proj   [t, 3072] * [3072, 768]
//
// Degenerate dimensions return without launching. Launch is asynchronous.
// Synchronise the stream before reading `c` on the host.
void gemm_forward(float* c, const float* a, const float* b, const float* bias,
                  int m, int n, int k, cudaStream_t stream);

// C[m, n] = A[m, k] * B[n, k]ᵀ + bias[n]
//
// The output head, and the one call that contracts along a stored row instead
// of a stored column. lm_head ties to wte, which is [n_vocab, d_model], so the
// weight arrives with its output axis first and nothing transposes it.
//
//   c     [m, n]  must not alias a, b or bias
//   a     [m, k]  activations, m = tokens
//   b     [n, k]  one contiguous row per output column
//   bias  [n]     optional, pass nullptr to skip it. The head has none.
//
// Same contract as gemm_forward otherwise.
void gemm_forward_bt(float* c, const float* a, const float* b, const float* bias,
                     int m, int n, int k, cudaStream_t stream);

}  // namespace nanoinfer
