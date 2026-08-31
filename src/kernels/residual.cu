// Residual add, naive elementwise.
//
// One add per 12 bytes moved, so the kernel is memory bound by a wide margin
// and a scalar grid-stride loop already sits near the bandwidth roofline. The
// remaining win is removing the launch entirely by folding the add into the
// epilogue of the GEMM that produces b.

#include "nanoinfer/kernels/residual.cuh"

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/device_ops.cuh"

namespace nanoinfer {
namespace {

constexpr int kBlock = 256;

// No __restrict__. out aliases a on every call in the block forward, and
// restrict would promise an absence of aliasing the caller does not provide.
__global__ void residual_kernel(float* out, const float* a, const float* b,
                                int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace

void residual_add(float* out, const float* a, const float* b, int n,
                  cudaStream_t stream) {
  if (n <= 0) return;

  residual_kernel<<<ceil_div(n, kBlock), kBlock, 0, stream>>>(out, a, b, n);
  CUDA_CHECK_LAUNCH();
}

}  // namespace nanoinfer
