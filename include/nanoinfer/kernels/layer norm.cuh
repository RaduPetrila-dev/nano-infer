// nano-infer: LayerNorm launcher.
//
// Headers under kernels/ declare launchers, never kernels. A __global__ function
// in a header would be compiled into every translation unit that includes it,
// and nothing outside the .cu needs to know the launch configuration.
#pragma once

#include <cuda_runtime.h>

namespace nanoinfer {

// y[r][c] = (x[r][c] - mean[r]) * rsqrt(var[r] + eps) * gamma[c] + beta[c]
//
// Normalisation runs over the last axis. mean and var are per row, computed over
// all `cols` elements of that row. Variance is the biased estimator, dividing by
// cols and not cols - 1, matching torch.nn.LayerNorm.
//
//   out    [rows, cols]  may alias in for an in-place normalise
//   in     [rows, cols]  rows = tokens in the batch, cols = d_model
//   gamma  [cols]        learned scale, shared across rows
//   beta   [cols]        learned shift, shared across rows
//   eps                  use kLayerNormEps for GPT-2
//
// Launch is asynchronous. Synchronise the stream before reading `out` on the
// host.
void layernorm_forward(float* out, const float* in, const float* gamma,
                       const float* beta, int rows, int cols, float eps,
                       cudaStream_t stream);

// Block size the launcher uses. Exposed so benchmarks can report it and tests
// can construct rows that exercise the tail path.
int layernorm_block_size(int cols);

}  // namespace nanoinfer
