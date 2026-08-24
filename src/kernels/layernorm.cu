// LayerNorm, naive version. One block per row, three passes over global memory.
// The three passes are deliberate: the optimised version caches the row in
// registers and reads once, and the gap is a number worth measuring.
// Design notes and the numerical reasoning live in docs/kernels.md.
#include "nanoinfer/kernels/layernorm.cuh"

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/device_ops.cuh"

namespace nanoinfer {
namespace {

constexpr int kReduceSlots = 32;  // one slot per warp at the 1024-thread maximum

// out may alias in. Each thread reads row_in[c] before writing row_out[c], and
// no thread reads a column another thread writes.
__global__ void layernorm_kernel(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta, int cols,
                                 float eps) {
  const int row = blockIdx.x;
  const float* row_in = in + static_cast<long long>(row) * cols;
  float* row_out = out + static_cast<long long>(row) * cols;

  __shared__ float reduce[kReduceSlots];

  // Threads with no element left must still reach block_reduce_sum. It contains
  // __syncthreads(), and a barrier some threads skip hangs the block.
  float partial_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    partial_sum += row_in[c];
  }

  // Biased estimator, matching torch.nn.LayerNorm. Dividing by cols - 1 gives a
  // 0.07% error at cols = 768 that fails parity and looks like nothing.
  const float mean =
      block_reduce_sum(partial_sum, reduce) / static_cast<float>(cols);

  // Squared deviations, not E[x^2] - E[x]^2. The one-pass formula cancels
  // catastrophically once the mean is large relative to the spread, which is
  // exactly what GPT-2's residual stream becomes by the later blocks. It passes
  // layer 0 and fails layer 11. See docs/kernels.md.
  float partial_var = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float deviation = row_in[c] - mean;
    partial_var += deviation * deviation;
  }

  const float variance =
      block_reduce_sum(partial_var, reduce) / static_cast<float>(cols);

  // Epsilon inside the square root, matching torch.nn.LayerNorm. rsqrtf is one
  // instruction but about 2 ulp, which eats half the 5e-7 elementwise budget.
  const float inv_std = 1.0f / sqrtf(variance + eps);

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row_out[c] = (row_in[c] - mean) * inv_std * gamma[c] + beta[c];
  }
}

}  // namespace

int layernorm_block_size(int cols) {
  // 256 divides 768 evenly at 3 elements per thread. 1024 leaves threads idle,
  // 128 halves the warps available to hide memory latency.
  if (cols >= 256) return 256;
  const int rounded = ceil_div(cols, kWarpSize) * kWarpSize;
  return rounded < kWarpSize ? kWarpSize : rounded;
}

void layernorm_forward(float* out, const float* in, const float* gamma,
                       const float* beta, int rows, int cols, float eps,
                       cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;

  // One block per row. Decode has one row, so one block runs on the whole
  // device and the kernel is pure latency. Fusing into the following GEMM or
  // capturing the step in a CUDA graph is the fix, not a different block size.
  const int threads = layernorm_block_size(cols);
  layernorm_kernel<<<rows, threads, 0, stream>>>(out, in, gamma, beta, cols, eps);
  CUDA_CHECK_LAUNCH();
}

}  // namespace nanoinfer
