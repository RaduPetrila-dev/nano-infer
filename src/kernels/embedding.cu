// Token and position embedding, naive gather.
//
// One block per token. The row is contiguous in both tables, so consecutive
// lanes touch consecutive addresses and every load coalesces. Indexing by
// element instead would scatter one row across many blocks and cost the same
// bandwidth for worse locality.

#include "nanoinfer/kernels/embedding.cuh"

#include <cassert>
#include <stdexcept>
#include <string>

#include "nanoinfer/cuda_utils.hpp"

namespace nanoinfer {
namespace {

constexpr int kBlock = 256;

__global__ void embedding_kernel(float* __restrict__ out,
                                 const int* __restrict__ ids,
                                 const float* __restrict__ wte,
                                 const float* __restrict__ wpe, int d_model,
                                 int pos_offset, int n_vocab) {
  const int token = blockIdx.x;
  const int id = ids[token];
  assert(id >= 0 && id < n_vocab);
  (void)n_vocab;  // NDEBUG drops the assert and with it the only use

  const float* wte_row = wte + static_cast<long long>(id) * d_model;
  const float* wpe_row =
      wpe + static_cast<long long>(pos_offset + token) * d_model;
  float* out_row = out + static_cast<long long>(token) * d_model;

  for (int c = threadIdx.x; c < d_model; c += blockDim.x) {
    out_row[c] = wte_row[c] + wpe_row[c];
  }
}

}  // namespace

void embedding_forward(float* out, const int* ids, const float* wte,
                       const float* wpe, int tokens, int d_model, int pos_offset,
                       int n_vocab, int n_ctx, cudaStream_t stream) {
  if (tokens <= 0 || d_model <= 0) return;

  // A position past n_ctx reads outside wpe and returns plausible garbage, so
  // it is worth a hard failure rather than a wrong logit.
  if (pos_offset < 0 || pos_offset + tokens > n_ctx) {
    throw std::runtime_error(
        "embedding: positions " + std::to_string(pos_offset) + ".." +
        std::to_string(pos_offset + tokens - 1) + " exceed context " +
        std::to_string(n_ctx));
  }

  // No reduction here, so a block wider than the row leaves threads idle
  // without a barrier hazard. A fixed block keeps the launcher trivial.
  embedding_kernel<<<tokens, kBlock, 0, stream>>>(out, ids, wte, wpe, d_model,
                                                  pos_offset, n_vocab);
  CUDA_CHECK_LAUNCH();
}

}  // namespace nanoinfer
