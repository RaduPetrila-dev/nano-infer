// nano-infer: token and position embedding launcher.
#pragma once

#include <cuda_runtime.h>

namespace nanoinfer {

// out[t][c] = wte[ids[t]][c] + wpe[pos_offset + t][c]
//
//   out         [tokens, d_model]
//   ids         [tokens]            device-side token ids, each below n_vocab
//   wte         [n_vocab, d_model]
//   wpe         [n_ctx, d_model]
//   pos_offset  absolute position of ids[0]. Zero during prefill, equal to the
//               number of cached tokens during decode.
//
// Throws std::runtime_error when the requested positions run past n_ctx. Ids
// are checked by a device assert, which release builds drop.
//
// Launch is asynchronous. Synchronise the stream before reading `out` on the
// host.
void embedding_forward(float* out, const int* ids, const float* wte,
                       const float* wpe, int tokens, int d_model, int pos_offset,
                       int n_vocab, int n_ctx, cudaStream_t stream);

}  // namespace nanoinfer
