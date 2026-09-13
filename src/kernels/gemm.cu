// GEMM, naive version. One thread per output element, each walking the whole
// reduction axis. No shared memory, no register blocking, no vectorised access.
//
// The point of this version is the baseline. Every later GEMM gets measured
// against it on the same card, and a speedup with no baseline is not a
// measurement. Design notes and the tolerance derivation live in docs/kernels.md.

#include "nanoinfer/kernels/gemm.cuh"

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/device_ops.cuh"
#include "nanoinfer/launch.cuh"

namespace nanoinfer {
namespace {

// Every pointer carries __restrict__, unlike the elementwise kernels. A GEMM
// cannot run in place: each output reads a whole row of A and a whole column of
// B, so writing C over either would race with reads from other blocks. The
// launcher forbids the aliasing outright, which makes the promise honest.
__global__ void gemm_kernel(float* __restrict__ c, const float* __restrict__ a,
                            const float* __restrict__ b,
                            const float* __restrict__ bias, int m, int n,
                            int k) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  // The grid rounds up to whole tiles, so the edge blocks carry threads with no
  // output. No barrier follows, so they leave immediately.
  if (row >= m || col >= n) return;

  const float* a_row = a + static_cast<long long>(row) * k;

  // Walk down the column of B by adding the row stride rather than recomputing
  // kk * n each step. Pointer arithmetic is 64-bit, so a weight larger than
  // 2^31 elements still indexes correctly.
  const float* b_col = b + col;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    // Explicit fmaf so the numerics do not depend on whether the compiler
    // contracts the multiply and the add. One rounding per term instead of two
    // also halves the accumulated error, which matters at k = 3072.
    acc = fmaf(a_row[kk], *b_col, acc);
    b_col += n;
  }

  // Bias last, outside the reduction. Seeding the accumulator with it instead
  // would put it at the head of a 3072-long dependency chain and round it 3072
  // times rather than once. The branch is uniform across the whole grid, so it
  // costs nothing.
  if (bias != nullptr) acc += bias[col];

  c[static_cast<long long>(row) * n + col] = acc;
}

}  // namespace

void gemm_forward(float* c, const float* a, const float* b, const float* bias,
                  int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return;

  const dim3 block(kGemmTileN, kGemmTileM);
  const dim3 grid(static_cast<unsigned>(ceil_div(n, kGemmTileN)),
                  static_cast<unsigned>(ceil_div(m, kGemmTileM)));

  gemm_kernel<<<grid, block, 0, stream>>>(c, a, b, bias, m, n, k);
  CUDA_CHECK_LAUNCH();
}

}  // namespace nanoinfer
