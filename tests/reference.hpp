// nano-infer tests: numerical comparison against reference activations.
//
// The output of this file is the thing you read a hundred times while debugging
// a kernel, so it reports where the error is, not only that there is one.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "npy.hpp"

namespace nanoinfer::test {

// CMake defines this to the absolute path of tests/data. Tests run from the
// build tree, so a relative path would break the moment you build out of source.
#ifndef NANOINFER_TEST_DATA_DIR
#define NANOINFER_TEST_DATA_DIR "tests/data"
#endif

// Exit with this and ctest records a skip rather than a failure. Reference data
// is gitignored, so a fresh clone must not report red tests before the dump has
// been generated.
constexpr int kSkipExitCode = 77;

inline std::string ref_path(const std::string& name) {
  return std::string(NANOINFER_TEST_DATA_DIR) + "/" + name + ".npy";
}

inline bool reference_available() {
  std::FILE* handle = std::fopen(ref_path("logits").c_str(), "rb");
  if (handle == nullptr) return false;
  std::fclose(handle);
  return true;
}

// Call at the top of any test that reads reference data.
inline int skip_without_reference() {
  if (reference_available()) return 0;
  std::printf(
      "SKIP no reference data in %s, run: python tools/dump_reference.py\n",
      NANOINFER_TEST_DATA_DIR);
  return kSkipExitCode;
}

// Tolerance defaults, chosen against fp32 accumulation error rather than taste.
//
// LayerNorm and GELU touch each element a handful of times, so error stays near
// one ulp. A GEMM with K = 768 accumulates 768 products per output, and fp32
// rounding error grows roughly with sqrt(K), giving about 28 * 2^-24, near 2e-6
// relative. Attention adds a softmax and a second GEMM on top. Full-model logits
// run all of that through 12 blocks.
//
// These are ceilings for a correct kernel, not targets. If a kernel needs a
// looser tolerance than the one below, the kernel is wrong.
struct Tolerance {
  float relative = 2e-5f;
  float absolute = 1e-6f;

  static Tolerance elementwise() { return {5e-7f, 1e-7f}; }  // layernorm, gelu
  static Tolerance gemm() { return {2e-5f, 1e-6f}; }
  static Tolerance attention() { return {5e-5f, 1e-6f}; }
  static Tolerance logits() { return {2e-4f, 1e-5f}; }
};

struct CompareResult {
  bool passed = false;
  std::size_t count = 0;
  std::size_t failures = 0;
  std::size_t non_finite = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double rms_error = 0.0;
  std::size_t worst_index = 0;
  float worst_actual = 0.0f;
  float worst_expected = 0.0f;
};

// The mixed criterion, the same one numpy.allclose uses:
//
//     |a - e| <= absolute + relative * |e|
//
// Pure relative error explodes near zero, and post-LayerNorm activations sit
// near zero constantly, so a correct kernel would fail on noise. Pure absolute
// error is scale-dependent, and logits reach magnitude 100 while normalised
// activations sit around 1, so one threshold cannot serve both. The sum handles
// each regime in the range where it makes sense.
inline CompareResult compare(const std::vector<float>& actual,
                             const std::vector<float>& expected,
                             Tolerance tol = {}) {
  CompareResult result;
  result.count = std::min(actual.size(), expected.size());

  double sum_sq = 0.0;
  double worst_score = -1.0;

  for (std::size_t i = 0; i < result.count; ++i) {
    const float a = actual[i];
    const float e = expected[i];

    if (!std::isfinite(a) || !std::isfinite(e)) {
      ++result.non_finite;
      ++result.failures;
      if (worst_score < 0.0) {
        result.worst_index = i;
        result.worst_actual = a;
        result.worst_expected = e;
        worst_score = 0.0;
      }
      continue;
    }

    const double abs_err = std::abs(static_cast<double>(a) - e);
    const double allowed = tol.absolute + tol.relative * std::abs(e);
    const double rel_err =
        std::abs(e) > 0.0 ? abs_err / std::abs(static_cast<double>(e)) : abs_err;

    sum_sq += abs_err * abs_err;
    result.max_abs_error = std::max(result.max_abs_error, abs_err);
    result.max_rel_error = std::max(result.max_rel_error, rel_err);

    if (abs_err > allowed) {
      ++result.failures;
      // Rank by how far past the budget the element is, not by raw error. A
      // large error on a large value is less interesting than a small error on
      // a small one, and the second kind is what points at a real bug.
      const double score = abs_err / allowed;
      if (score > worst_score) {
        worst_score = score;
        result.worst_index = i;
        result.worst_actual = a;
        result.worst_expected = e;
      }
    }
  }

  result.rms_error =
      result.count > 0 ? std::sqrt(sum_sq / static_cast<double>(result.count)) : 0.0;
  result.passed = result.failures == 0 && actual.size() == expected.size();
  return result;
}

// Prints one line on success and a diagnostic block on failure. The failing
// index is reported in row and column form as well as flat, because a kernel bug
// almost always clusters on one axis. All of row 0 wrong means the first block
// is broken. Column 0 of every row wrong means lane 0 is. A scattered handful
// usually means a race.
inline bool report(const std::string& label, const CompareResult& r,
                   std::size_t cols = 0) {
  if (r.passed) {
    std::printf("PASS %-28s n=%zu  max_abs=%.3e  max_rel=%.3e\n", label.c_str(),
                r.count, r.max_abs_error, r.max_rel_error);
    return true;
  }

  std::printf("FAIL %-28s n=%zu  %zu/%zu elements out of tolerance\n",
              label.c_str(), r.count, r.failures, r.count);
  std::printf("     max_abs=%.3e  max_rel=%.3e  rms=%.3e\n", r.max_abs_error,
              r.max_rel_error, r.rms_error);
  if (r.non_finite > 0) {
    std::printf("     %zu non-finite values, check for a division by zero or an "
                "uninitialised read\n",
                r.non_finite);
  }
  std::printf("     worst at flat index %zu", r.worst_index);
  if (cols > 0) {
    std::printf(" (row %zu, col %zu)", r.worst_index / cols, r.worst_index % cols);
  }
  std::printf(": got %.9g, expected %.9g\n", static_cast<double>(r.worst_actual),
              static_cast<double>(r.worst_expected));
  return false;
}

// Copies a device buffer back and compares it against a .npy reference.
inline bool check_device(const std::string& label, const float* device_ptr,
                         const std::string& reference_path, Tolerance tol = {}) {
  const NpyArray reference = load_npy(reference_path);

  std::vector<float> host(reference.size());
  CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const CompareResult result = compare(host, reference.data, tol);
  return report(label, result, reference.cols());
}

// Same, for a kernel already staged into host memory.
inline bool check_host(const std::string& label, const std::vector<float>& actual,
                       const std::string& reference_path, Tolerance tol = {}) {
  const NpyArray reference = load_npy(reference_path);

  if (actual.size() != reference.size()) {
    std::printf("FAIL %-28s size %zu, reference %s has %zu\n", label.c_str(),
                actual.size(), reference.shape_string().c_str(), reference.size());
    return false;
  }

  const CompareResult result = compare(actual, reference.data, tol);
  return report(label, result, reference.cols());
}

// Small tracker so a test file can run several checks and exit once.
class TestRun {
 public:
  void add(bool passed) {
    ++total_;
    if (!passed) ++failed_;
  }

  int finish(const std::string& name) const {
    if (failed_ == 0) {
      std::printf("PASS %s (%d checks)\n", name.c_str(), total_);
      return 0;
    }
    std::printf("FAIL %s (%d of %d checks failed)\n", name.c_str(), failed_,
                total_);
    return 1;
  }

 private:
  int total_ = 0;
  int failed_ = 0;
};

}  // namespace nanoinfer::test
