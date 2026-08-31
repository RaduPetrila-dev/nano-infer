// nano-infer: GELU launcher.
#pragma once

#include <cuda_runtime.h>

namespace nanoinfer {

// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The tanh approximation, not the erf definition. HuggingFace GPT-2 sets
// activation_function to "gelu_new", and the two forms differ by up to 4.7e-4
// absolute near x = 2.7, three orders above the elementwise tolerance.
//
//   out  [n]  may alias in
//   in   [n]  MLP intermediate, n = tokens * 4 * d_model
//
// Launch is asynchronous. Synchronise the stream before reading `out` on the
// host.
void gelu_forward(float* out, const float* in, int n, cudaStream_t stream);

}  // namespace nanoinfer
