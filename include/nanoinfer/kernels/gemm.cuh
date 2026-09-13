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
// The output head is the one call that contracts against a stored row rather
// than a column, since lm_head ties to wte and wte is [n_vocab, d_model]. That
// needs a transposed-B variant and arrives with the end-to-end logit test.
//
// Degenerate dimensions return without launching. Launch is asynchronous.
// Synchronise the stream before reading `c` on the host.
void gemm_forward(float* c, const float* a, const float* b, const float* bias,
                  int m, int n, int k, cudaStream_t stream);

}  // namespace nanoinfer
