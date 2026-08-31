// GELU, naive elementwise. One thread per element, grid-stride so the launcher
// never has to reason about grid limits.

#include "nanoinfer/kernels/gelu.cuh"

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/device_ops.cuh"

namespace nanoinfer {
namespace {

constexpr int kBlock = 256;

// Constants from HuggingFace NewGELUActivation.
constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2 / pi)
constexpr float kBeta = 0.044715f;

__device__ __forceinline__ float gelu_tanh(float x) {
  // Explicit fmaf so the polynomial does not depend on whether the compiler
  // contracts the multiply and add.
  const float inner = kAlpha * fmaf(kBeta * x * x, x, x);

  // Below x = -5.16 the fp32 sum 1 + tanhf collapses to exactly zero and the
  // result is a signed zero. That is the correct limit and torch agrees.
  return 0.5f * x * (1.0f + tanhf(inner));
}

// No __restrict__ on either pointer. In-place GELU is the normal call, and
// restrict would promise the compiler an absence of aliasing the caller does
// not provide.
__global__ void gelu_kernel(float* out, const float* in, int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = gelu_tanh(in[i]);
  }
}

}  // namespace

void gelu_forward(float* out, const float* in, int n, cudaStream_t stream) {
  if (n <= 0) return;

  gelu_kernel<<<ceil_div(n, kBlock), kBlock, 0, stream>>>(out, in, n);
  CUDA_CHECK_LAUNCH();
}

}  // namespace nanoinfer
