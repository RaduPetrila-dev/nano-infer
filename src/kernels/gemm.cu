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

// One warp per output element, lanes striding the reduction axis. Both operands
// are read along k here, so a warp's 32 addresses are contiguous in each. The
// thread-per-output mapping above would read b with a stride of k and split
// every load into 32 transactions.
//
// The reduction is a per-lane strided sum followed by a butterfly, so it rounds
// at least as well as the sequential pass the tolerance floor is derived from.
__global__ void gemm_bt_kernel(float* __restrict__ c, const float* __restrict__ a,
                               const float* __restrict__ b,
                               const float* __restrict__ bias, int n, int k) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWarpSize;
  const int warps = static_cast<int>(blockDim.x) / kWarpSize;
  const int col = static_cast<int>(blockIdx.x) * warps + tid / kWarpSize;
  const int row = static_cast<int>(blockIdx.y);

  // Warp-uniform, so no lane leaves a shuffle short-handed.
  if (col >= n) return;

  const float* a_row = a + static_cast<long long>(row) * k;
  const float* b_row = b + static_cast<long long>(col) * k;

  float acc = 0.0f;
  for (int kk = lane; kk < k; kk += kWarpSize) {
    acc = fmaf(a_row[kk], b_row[kk], acc);
  }
  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    if (bias != nullptr) acc += bias[col];
    c[static_cast<long long>(row) * n + col] = acc;
  }
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

void gemm_forward_bt(float* c, const float* a, const float* b, const float* bias,
                     int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return;

  const int warps = kBlockThreads / kWarpSize;
  const dim3 grid(static_cast<unsigned>(ceil_div(n, warps)),
                  static_cast<unsigned>(m));

  gemm_bt_kernel<<<grid, kBlockThreads, 0, stream>>>(c, a, b, bias, n, k);
  CUDA_CHECK_LAUNCH();
}

}  // namespace nanoinfer
