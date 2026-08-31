// nano-infer: residual add launcher.
#pragma once

#include <cuda_runtime.h>

namespace nanoinfer {

// out[i] = a[i] + b[i]
//
// out may alias a or b. In the block forward it always does: the residual
// stream is updated in place after each sub-layer.
//
//   out  [n]
//   a    [n]  residual stream entering the sub-layer
//   b    [n]  sub-layer output
//
// Launch is asynchronous. Synchronise the stream before reading `out` on the
// host.
void residual_add(float* out, const float* a, const float* b, int n,
                  cudaStream_t stream);

}  // namespace nanoinfer
