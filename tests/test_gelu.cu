// nano-infer: GELU tests.
//
// Two tiers, the same shape as the LayerNorm tests. Synthetic cases check the
// kernel against a double-precision CPU oracle. The parity case pushes the real
// HuggingFace layer 0 MLP intermediate through the kernel and compares against
// the tensor entering c_proj, which is post-GELU by construction.
//
// The negative tail is the case worth reading. gelu(-4) is -7.0e-5 while the
// input is -4, so the fp32 error floor there is three orders above the value
// itself in relative terms. A relative-only tolerance rejects a correct kernel.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/kernels/gelu.cuh"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

// The tanh form in double. Matching HuggingFace NewGELUActivation is the point:
// the erf definition peaks 4.7e-4 away from this one at |x| = 2.7, which misses
// the tolerance below by 99x.
double gelu_ref(double x) {
  constexpr double kAlpha = 0.7978845608028654;  // sqrt(2 / pi)
  constexpr double kBeta = 0.044715;
  return 0.5 * x * (1.0 + std::tanh(kAlpha * (x + kBeta * x * x * x)));
}

std::vector<float> gelu_cpu(const std::vector<float>& in) {
  std::vector<float> out(in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<float>(gelu_ref(static_cast<double>(in[i])));
  }
  return out;
}

// The floor an fp32 GELU can reach against a double oracle.
//
// y = 0.5 * x * (1 + tanh(u)). tanhf carries about 2 ulp, so the absolute error
// is near 0.5 * |x| * 2 * 2^-24, which reduces to |x| * 2^-24. The tolerance
// below keeps an order of magnitude above that.
//
// The relative term stays at the elementwise default and does no work in the
// tail: 1 + tanh(u) collapses toward zero there while |x| stays large, so the
// relative error of a correct kernel is unbounded and only the absolute term is
// meaningful. At x = -4 the floor is 2.4e-7 absolute against a value of 7.0e-5.
Tolerance gelu_floor(const std::vector<float>& in) {
  constexpr float kEps = 5.96e-8f;  // 2^-24
  float max_abs = 0.0f;
  for (float v : in) max_abs = std::max(max_abs, std::fabs(v));

  Tolerance tol;
  tol.relative = 5e-7f;
  tol.absolute = 10.0f * max_abs * kEps;
  return tol;
}

std::vector<float> run_kernel(const std::vector<float>& in) {
  DeviceBuffer<float> d_in(in.size());
  DeviceBuffer<float> d_out(in.size());
  CUDA_CHECK(cudaMemcpy(d_in.get(), in.data(), d_in.bytes(),
                        cudaMemcpyHostToDevice));

  CudaStream stream;
  gelu_forward(d_out.get(), d_in.get(), static_cast<int>(in.size()),
               stream.get());
  stream.sync();

  std::vector<float> out(in.size());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(),
                        cudaMemcpyDeviceToHost));
  return out;
}

// Same input, one buffer. The launcher documents this as legal and the block
// forward relies on it.
std::vector<float> run_kernel_in_place(const std::vector<float>& in) {
  DeviceBuffer<float> d(in.size());
  CUDA_CHECK(cudaMemcpy(d.get(), in.data(), d.bytes(), cudaMemcpyHostToDevice));

  CudaStream stream;
  gelu_forward(d.get(), d.get(), static_cast<int>(in.size()), stream.get());
  stream.sync();

  std::vector<float> out(in.size());
  CUDA_CHECK(cudaMemcpy(out.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost));
  return out;
}

std::vector<float> linspace(std::size_t n, float lo, float hi) {
  std::vector<float> out(n);
  const float step = (hi - lo) / static_cast<float>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = lo + step * static_cast<float>(i);
  }
  return out;
}

std::vector<float> gaussian(std::size_t n, float spread, unsigned seed) {
  std::mt19937 rng(seed);  // fixed, a flaky numerical test teaches you to ignore it
  std::normal_distribution<float> noise(0.0f, spread);
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = noise(rng);
  return out;
}

bool check(const std::string& name, const std::vector<float>& in) {
  return report(name, compare(run_kernel(in), gelu_cpu(in), gelu_floor(in)));
}

bool run_parity() {
  const NpyArray input = load_npy(ref_path("h.0.mlp.fc.out"));
  return check_host("hf_parity_h0_mlp_gelu", run_kernel(input.data),
                    ref_path("h.0.mlp.proj.in"), gelu_floor(input.data));
}

}  // namespace

int main() {
  TestRun run;

  // Prefill shape of the MLP intermediate, 4 * d_model wide.
  run.add(check("mlp_8x3072", gaussian(8 * 3072, 2.0f, 1234)));
  // Decode shape. One row, so the grid shrinks to 12 blocks.
  run.add(check("decode_1x3072", gaussian(3072, 2.0f, 5678)));
  // Dense sweep across the region where the curve bends and where the tanh and
  // erf forms disagree most.
  run.add(check("sweep_pm8", linspace(4096, -8.0f, 8.0f)));
  // The cancellation region. 1 + tanh(u) reaches 3.5e-5 at x = -4 and flushes to
  // exactly zero below x = -5.16.
  run.add(check("negative_tail", linspace(2048, -12.0f, -3.0f)));
  // Saturation. Large positive returns x unchanged, large negative returns a
  // signed zero, and neither produces a NaN.
  run.add(check("saturation_pm40", linspace(1024, -40.0f, 40.0f)));
  // n not a multiple of the block size. Exercises the grid-stride tail.
  run.add(check("odd_n_1023", gaussian(1023, 2.0f, 99)));

  const std::vector<float> alias_input = gaussian(4096, 2.0f, 4242);
  run.add(report("in_place_4096",
                 compare(run_kernel_in_place(alias_input), gelu_cpu(alias_input),
                         gelu_floor(alias_input))));

  if (reference_available()) {
    run.add(run_parity());
  } else {
    std::printf("SKIP hf_parity_h0_mlp_gelu, run: python tools/dump_reference.py\n");
  }

  return run.finish("test_gelu");
}
