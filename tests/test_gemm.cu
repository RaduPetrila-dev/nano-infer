// nano-infer: GEMM tests.
//
// Synthetic cases check the kernel against a double-precision oracle at the four
// shapes a 124M block issues. The parity cases push real layer 0 activations
// through real layer 0 weights, which is the only check that catches a misread
// of the weight layout.
//
// The absolute tolerance is derived, not taken from Tolerance::gemm(). See
// gemm_floor below and the GEMM section of docs/kernels.md.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/kernels/gemm.cuh"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

constexpr double kEps = 5.9604644775390625e-8;  // 2^-24

// Oracle values and the error floor, from one pass.
//
// A dot product accumulates error in proportion to its partial sums, not its
// result. Step j rounds against S_j and the roundings add as a random walk:
//
//     floor = eps * sqrt( sum_j S_j^2 )
//
// Measured against a sequential fmaf reference, the real error lands between 0.4
// and 1.2 times this, so the 10x below is a margin. The flat 1e-6 in
// Tolerance::gemm() fails five of the ten shapes here, worst by 18x.
struct Oracle {
  std::vector<float> values;
  double floor = 0.0;  // absolute, worst output in the result
};

Oracle gemm_cpu(const std::vector<float>& a, const std::vector<float>& b,
                const std::vector<float>* bias, int m, int n, int k) {
  Oracle oracle;
  oracle.values.resize(static_cast<std::size_t>(m) * n);

  // Accumulate a whole output row at a time. The reduction order per element is
  // still k ascending, matching the kernel, and the inner loop walks b
  // sequentially instead of striding by n and missing every cache line.
  std::vector<double> acc(static_cast<std::size_t>(n));
  std::vector<double> partial_sq(static_cast<std::size_t>(n));

  double worst_sq = 0.0;
  for (int row = 0; row < m; ++row) {
    std::fill(acc.begin(), acc.end(), 0.0);
    std::fill(partial_sq.begin(), partial_sq.end(), 0.0);

    for (int kk = 0; kk < k; ++kk) {
      const double a_val = a[static_cast<std::size_t>(row) * k + kk];
      const float* b_row = b.data() + static_cast<std::size_t>(kk) * n;
      for (int col = 0; col < n; ++col) {
        acc[static_cast<std::size_t>(col)] += a_val * b_row[col];
        const double partial = acc[static_cast<std::size_t>(col)];
        partial_sq[static_cast<std::size_t>(col)] += partial * partial;
      }
    }

    for (int col = 0; col < n; ++col) {
      const std::size_t c = static_cast<std::size_t>(col);
      double value = acc[c];
      if (bias != nullptr) value += (*bias)[c];
      oracle.values[static_cast<std::size_t>(row) * n + c] =
          static_cast<float>(value);
      worst_sq = std::max(worst_sq, partial_sq[c]);
    }
  }

  oracle.floor = kEps * std::sqrt(worst_sq);
  return oracle;
}

// The relative ceiling from reference.hpp, absolute term replaced by the derived
// floor. Taking the worst output's floor for the whole comparison only loosens
// elements whose own partial sums were large, and those carry a large error too.
Tolerance gemm_tolerance(const Oracle& oracle) {
  Tolerance tol = Tolerance::gemm();
  tol.absolute = std::max(tol.absolute, static_cast<float>(10.0 * oracle.floor));
  return tol;
}

// Both launchers share a signature, so one runner serves both and the cases
// below differ only in how b is laid out.
using Launcher = void (*)(float*, const float*, const float*, const float*, int,
                          int, int, cudaStream_t);

std::vector<float> run_kernel(Launcher launch, const std::vector<float>& a,
                              const std::vector<float>& b,
                              const std::vector<float>* bias, int m, int n,
                              int k) {
  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_c(static_cast<std::size_t>(m) * n);
  DeviceBuffer<float> d_bias;
  if (bias != nullptr) d_bias.allocate(bias->size());

  CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), d_a.bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), d_b.bytes(), cudaMemcpyHostToDevice));
  if (bias != nullptr) {
    CUDA_CHECK(cudaMemcpy(d_bias.get(), bias->data(), d_bias.bytes(),
                          cudaMemcpyHostToDevice));
  }

  // Poison, so an edge tile that writes nothing fails instead of inheriting the
  // previous case's answer.
  CUDA_CHECK(cudaMemset(d_c.get(), 0x7f, d_c.bytes()));

  CudaStream stream;
  launch(d_c.get(), d_a.get(), d_b.get(),
         bias != nullptr ? d_bias.get() : nullptr, m, n, k, stream.get());
  stream.sync();

  std::vector<float> out(d_c.count());
  CUDA_CHECK(cudaMemcpy(out.data(), d_c.get(), d_c.bytes(),
                        cudaMemcpyDeviceToHost));
  return out;
}

std::vector<float> gaussian(std::size_t n, float spread, unsigned seed) {
  std::mt19937 rng(seed);  // fixed, a flaky numerical test teaches you to ignore it
  std::normal_distribution<float> noise(0.0f, spread);
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = noise(rng);
  return out;
}

struct Case {
  std::string name;
  int m;
  int n;
  int k;
  float a_spread;
  float b_spread;
  bool with_bias;
};

bool run_case(const Case& c) {
  const std::vector<float> a =
      gaussian(static_cast<std::size_t>(c.m) * c.k, c.a_spread, 1000u);
  const std::vector<float> b =
      gaussian(static_cast<std::size_t>(c.k) * c.n, c.b_spread, 2000u);
  const std::vector<float> bias =
      gaussian(static_cast<std::size_t>(c.n), 0.05f, 3000u);
  const std::vector<float>* bias_ptr = c.with_bias ? &bias : nullptr;

  const Oracle oracle = gemm_cpu(a, b, bias_ptr, c.m, c.n, c.k);
  const std::vector<float> actual =
      run_kernel(gemm_forward, a, b, bias_ptr, c.m, c.n, c.k);

  return report(c.name, compare(actual, oracle.values, gemm_tolerance(oracle)),
                static_cast<std::size_t>(c.n));
}

// b arrives as [n, k] for the transposed launcher, one contiguous row per output
// column. The oracle wants [k, n], so the test flips it on the host. A transpose
// in the test cannot hide one in the kernel, since the agreement case below runs
// both launchers over the same numbers in their own layouts.
std::vector<float> transpose(const std::vector<float>& in, int rows, int cols) {
  std::vector<float> out(in.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(c) * rows + r] =
          in[static_cast<std::size_t>(r) * cols + c];
    }
  }
  return out;
}

bool run_bt_case(const Case& c) {
  const std::vector<float> a =
      gaussian(static_cast<std::size_t>(c.m) * c.k, c.a_spread, 1000u);
  const std::vector<float> b =
      gaussian(static_cast<std::size_t>(c.n) * c.k, c.b_spread, 2000u);
  const std::vector<float> bias =
      gaussian(static_cast<std::size_t>(c.n), 0.05f, 3000u);
  const std::vector<float>* bias_ptr = c.with_bias ? &bias : nullptr;

  const Oracle oracle =
      gemm_cpu(a, transpose(b, c.n, c.k), bias_ptr, c.m, c.n, c.k);
  const std::vector<float> actual =
      run_kernel(gemm_forward_bt, a, b, bias_ptr, c.m, c.n, c.k);

  return report(c.name, compare(actual, oracle.values, gemm_tolerance(oracle)),
                static_cast<std::size_t>(c.n));
}

// The transposed launcher against the one the parity cases already cover, over
// the same numbers in both layouts. Everything else here checks a kernel against
// an oracle, which shares no code with either kernel but also shares no layout
// convention with the checkpoint.
bool run_bt_agreement() {
  const int m = 8;
  const int n = 768;
  const int k = 768;

  const std::vector<float> a = gaussian(static_cast<std::size_t>(m) * k, 1.0f, 11u);
  const std::vector<float> b_kn =
      gaussian(static_cast<std::size_t>(k) * n, 1.0f, 22u);
  const std::vector<float> b_nk = transpose(b_kn, k, n);

  const Oracle oracle = gemm_cpu(a, b_kn, nullptr, m, n, k);
  const std::vector<float> straight =
      run_kernel(gemm_forward, a, b_kn, nullptr, m, n, k);
  const std::vector<float> transposed =
      run_kernel(gemm_forward_bt, a, b_nk, nullptr, m, n, k);

  const Tolerance tol = gemm_tolerance(oracle);
  bool ok = report("bt_matches_oracle", compare(transposed, oracle.values, tol),
                   static_cast<std::size_t>(n));
  ok &= report("bt_matches_gemm", compare(transposed, straight, tol),
               static_cast<std::size_t>(n));
  return ok;
}

bool has_reference(const std::string& name) {
  std::FILE* handle = std::fopen(ref_path(name).c_str(), "rb");
  if (handle == nullptr) return false;
  std::fclose(handle);
  return true;
}

// One projection of layer 0 against the Conv1D that produced it. A synthetic
// case cannot catch a transpose, since it reads back whatever layout it wrote,
// so this is the only check on the layout convention.
bool run_parity(const std::string& module) {
  const NpyArray input = load_npy(ref_path(module + ".in"));
  const NpyArray weight = load_npy(ref_path(module + ".w"));
  const NpyArray bias = load_npy(ref_path(module + ".b"));

  const int m = static_cast<int>(input.rows());
  const int k = static_cast<int>(input.cols());
  const int n = static_cast<int>(weight.cols());

  if (static_cast<int>(weight.rows()) != k) {
    std::printf("FAIL %-28s weight is %s, expected %d rows\n", module.c_str(),
                weight.shape_string().c_str(), k);
    return false;
  }

  const Oracle oracle = gemm_cpu(input.data, weight.data, &bias.data, m, n, k);
  const std::vector<float> actual =
      run_kernel(gemm_forward, input.data, weight.data, &bias.data, m, n, k);

  // The reference is fp32 out of a blocked CPU GEMM, which sums more accurately
  // than a single sequential pass, so its own error sits inside the same floor.
  return check_host("hf_parity_" + module, actual, ref_path(module + ".out"),
                    gemm_tolerance(oracle));
}

// An empty batch is not a programming error. It must leave no sticky error
// behind either.
bool run_degenerate() {
  cudaGetLastError();
  CudaStream stream;

  const Launcher launchers[] = {gemm_forward, gemm_forward_bt};
  for (Launcher launch : launchers) {
    launch(nullptr, nullptr, nullptr, nullptr, 0, 8, 8, stream.get());
    launch(nullptr, nullptr, nullptr, nullptr, 8, 0, 8, stream.get());
    launch(nullptr, nullptr, nullptr, nullptr, 8, 8, 0, stream.get());
    launch(nullptr, nullptr, nullptr, nullptr, -1, 8, 8, stream.get());
  }

  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    std::printf("FAIL %-28s left %s set\n", "degenerate_dims",
                cudaGetErrorString(status));
    return false;
  }
  std::printf("PASS %-28s no launch\n", "degenerate_dims");
  return true;
}

}  // namespace

int main() {
  TestRun run;

  const Case cases[] = {
      // The four shapes a block issues, at a prefill of 8 tokens.
      {"qkv_8x2304x768", 8, 2304, 768, 1.0f, 1.0f, true},
      {"attn_proj_8x768x768", 8, 768, 768, 1.0f, 1.0f, true},
      {"mlp_fc_8x3072x768", 8, 3072, 768, 1.0f, 1.0f, true},
      // The only call that contracts over 3072, so four times the reduction
      // depth and twice the accumulated error.
      {"mlp_proj_8x768x3072", 8, 768, 3072, 1.0f, 1.0f, true},
      // Decode. Seven of every eight threads in the single tile row idle.
      {"decode_1x2304x768", 1, 2304, 768, 1.0f, 1.0f, true},
      // Realistic magnitudes, which shrink the result and the floor together.
      {"gpt2_scales_8x2304x768", 8, 2304, 768, 1.0f, 0.02f, true},
      // Every dimension off the tile, so both guards fire.
      {"ragged_5x77x33", 5, 77, 33, 1.0f, 1.0f, true},
      // n below one tile. The whole grid is a single partial block.
      {"narrow_n_8x7x768", 8, 7, 768, 1.0f, 1.0f, true},
      // A reduction of length one, the shortest legal loop.
      {"k_is_one_4x64x1", 4, 64, 1, 1.0f, 1.0f, true},
      // bias == nullptr, the path the output head will take.
      {"no_bias_8x768x768", 8, 768, 768, 1.0f, 1.0f, false},
  };

  for (const Case& c : cases) run.add(run_case(c));

  // The output head, which contracts along a stored row. The real head is
  // n = 50257 and the end-to-end test covers that shape against HuggingFace.
  const Case bt_cases[] = {
      {"head_4x5000x768", 4, 5000, 768, 1.0f, 0.02f, false},
      // Every dimension off the warp and the block tile.
      {"bt_ragged_5x77x33", 5, 77, 33, 1.0f, 1.0f, true},
      // k below one warp, so most lanes carry nothing into the shuffle.
      {"bt_narrow_k_3x64x7", 3, 64, 7, 1.0f, 1.0f, true},
      // The shortest legal reduction.
      {"bt_k_is_one_4x64x1", 4, 64, 1, 1.0f, 1.0f, false},
  };
  for (const Case& c : bt_cases) run.add(run_bt_case(c));
  run.add(run_bt_agreement());

  run.add(run_degenerate());

  // Parity needs the Conv1D weights, which the dump only writes when asked:
  //   python tools/dump_reference.py --dump-weights
  const char* modules[] = {"h.0.attn.qkv", "h.0.attn.proj", "h.0.mlp.fc",
                           "h.0.mlp.proj"};
  for (const char* module : modules) {
    if (has_reference(std::string(module) + ".w")) {
      run.add(run_parity(module));
    } else {
      std::printf("SKIP hf_parity_%s, run: python tools/dump_reference.py "
                  "--dump-weights\n", module);
    }
  }

  return run.finish("test_gemm");
}
