// nano-infer: LayerNorm tests.
//
// Two tiers. Synthetic cases check the kernel against a double-precision CPU
// oracle and cover the shapes and magnitudes the reference dump does not reach.
// The parity case checks it against the real HuggingFace layer 0 tensor.
//
// The synthetic tier matters more than it looks. Layer 0 activations are small
// and well behaved, so a kernel using the unstable variance formula passes
// parity and still breaks at layer 11. The large-mean case below is what catches
// that.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/kernels/layernorm.cuh"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

// Oracle in double precision. The kernel runs in fp32, so the reference has to
// carry more precision than the thing it judges, or you are comparing two
// implementations of the same rounding error.
std::vector<float> layernorm_cpu(const std::vector<float>& in,
                                 const std::vector<float>& gamma,
                                 const std::vector<float>& beta, int rows,
                                 int cols, float eps) {
  std::vector<float> out(in.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = in.data() + static_cast<std::size_t>(r) * cols;

    double mean = 0.0;
    for (int c = 0; c < cols; ++c) mean += row[c];
    mean /= cols;

    double var = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double d = static_cast<double>(row[c]) - mean;
      var += d * d;
    }
    var /= cols;

    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));
    for (int c = 0; c < cols; ++c) {
      const double norm = (static_cast<double>(row[c]) - mean) * inv_std;
      out[static_cast<std::size_t>(r) * cols + c] =
          static_cast<float>(norm * gamma[c] + beta[c]);
    }
  }
  return out;
}

// Uploads, launches, downloads. Keeps each case to a few lines.
std::vector<float> run_kernel(const std::vector<float>& in,
                              const std::vector<float>& gamma,
                              const std::vector<float>& beta, int rows, int cols,
                              float eps) {
  DeviceBuffer<float> d_in(in.size());
  DeviceBuffer<float> d_out(in.size());
  DeviceBuffer<float> d_gamma(gamma.size());
  DeviceBuffer<float> d_beta(beta.size());

  CUDA_CHECK(cudaMemcpy(d_in.get(), in.data(), d_in.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma.get(), gamma.data(), d_gamma.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta.get(), beta.data(), d_beta.bytes(),
                        cudaMemcpyHostToDevice));

  CudaStream stream;
  layernorm_forward(d_out.get(), d_in.get(), d_gamma.get(), d_beta.get(), rows,
                    cols, eps, stream.get());
  stream.sync();

  std::vector<float> out(in.size());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(),
                        cudaMemcpyDeviceToHost));
  return out;
}

struct Case {
  std::string name;
  int rows;
  int cols;
  float centre;  // mean the input is shifted to
  float spread;
  Tolerance tol;
};

// The floor an fp32 kernel can reach against a double-precision oracle.
//
// Summing N values of magnitude |centre| in fp32 leaves the sum with an absolute
// error near eps * |centre| * sqrt(N), where eps is 2^-24. Dividing by N gives
// the error in the mean, and normalising divides by the spread, so the error
// lands in the output as
//
//     eps * |centre| * sqrt(N) / spread
//
// At centre 4000, spread 1 and N 768 that is about 6.6e-3. No fp32
// implementation beats it, however the variance is computed, so demanding the
// 5e-7 elementwise budget here would fail a correct kernel. The tolerance below
// keeps roughly an order of magnitude of margin above the floor and still sits
// two to three orders below what the unstable variance formula produces.
Tolerance fp32_floor(float centre, float spread, int cols) {
  constexpr float kEps = 5.96e-8f;  // 2^-24
  const float floor_error =
      kEps * std::fabs(centre) * std::sqrt(static_cast<float>(cols)) / spread;
  Tolerance tol;
  tol.relative = 1e-5f;
  tol.absolute = 10.0f * floor_error;
  return tol;
}

bool run_case(const Case& c) {
  // Fixed seed. A flaky numerical test is worse than no test, because you stop
  // trusting the ones that are telling the truth.
  std::mt19937 rng(1234 + c.rows * 31 + c.cols);
  std::normal_distribution<float> noise(0.0f, c.spread);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

  const std::size_t n = static_cast<std::size_t>(c.rows) * c.cols;
  std::vector<float> in(n);
  for (std::size_t i = 0; i < n; ++i) in[i] = c.centre + noise(rng);

  std::vector<float> gamma(c.cols);
  std::vector<float> beta(c.cols);
  for (int i = 0; i < c.cols; ++i) {
    // Centred on 1 and 0, the ranges trained LayerNorm parameters actually
    // occupy. Uniform noise would not catch a swapped gamma and beta.
    gamma[i] = 1.0f + 0.1f * unit(rng);
    beta[i] = 0.05f * unit(rng);
  }

  const std::vector<float> expected =
      layernorm_cpu(in, gamma, beta, c.rows, c.cols, kLayerNormEps);
  const std::vector<float> actual =
      run_kernel(in, gamma, beta, c.rows, c.cols, kLayerNormEps);

  return report(c.name, compare(actual, expected, c.tol),
                static_cast<std::size_t>(c.cols));
}

bool run_parity() {
  const NpyArray input = load_npy(ref_path("h.0.ln_1.in"));
  const NpyArray gamma = load_npy(ref_path("h.0.ln_1.gamma"));
  const NpyArray beta = load_npy(ref_path("h.0.ln_1.beta"));

  const int rows = static_cast<int>(input.rows());
  const int cols = static_cast<int>(input.cols());

  const std::vector<float> actual =
      run_kernel(input.data, gamma.data, beta.data, rows, cols, kLayerNormEps);

  return check_host("hf_parity_h0_ln1", actual, ref_path("h.0.ln_1.out"),
                    Tolerance::elementwise());
}

}  // namespace

int main() {
  TestRun run;

  const Case cases[] = {
      // Prefill shape. Many blocks, the GPU fills.
      {"prefill_8x768", 8, 768, 0.0f, 1.0f, Tolerance::elementwise()},
      // Decode shape. One row means one block on the whole device. Correctness
      // is the same, occupancy is not.
      {"decode_1x768", 1, 768, 0.0f, 1.0f, Tolerance::elementwise()},
      // cols not a multiple of the warp size. Exercises the grid-stride tail and
      // the identity contribution from threads with no element left. Get this
      // wrong and the block hangs or the sum comes up short.
      {"ragged_3x100", 3, 100, 0.0f, 1.0f, Tolerance::elementwise()},
      // cols smaller than one warp. Checks the block size floor.
      {"narrow_4x17", 4, 17, 0.0f, 1.0f, Tolerance::elementwise()},
      // The cancellation case. Mean near 4000, deviations near 1, so E[x^2] is
      // around 1.6e7 while the variance is around 1. A correct kernel lands near
      // 2e-3. E[x^2] - E[x]^2 lands near 2.0 and produces negative variances,
      // which come back as NaN once you take the square root.
      {"large_mean_4x768", 4, 768, 4000.0f, 1.0f, fp32_floor(4000.0f, 1.0f, 768)},
      // Same trap, milder, closer to what layer 11 actually looks like. Correct
      // lands near 1e-4, the unstable formula near 6e-2.
      {"outlier_scale_4x768", 4, 768, 300.0f, 2.0f, fp32_floor(300.0f, 2.0f, 768)},
  };

  for (const Case& c : cases) run.add(run_case(c));

  if (reference_available()) {
    run.add(run_parity());
  } else {
    std::printf("SKIP hf_parity_h0_ln1, run: python tools/dump_reference.py\n");
  }

  return run.finish("test_layernorm");
}
