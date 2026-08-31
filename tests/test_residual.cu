// nano-infer: residual add tests.
//
// The arithmetic is one fp32 add, so the cases cover the two things worth
// getting wrong: the grid-stride tail when n is not a multiple of the block,
// and in-place aliasing, which is how the block forward always calls it.

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/kernels/residual.cuh"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

enum class Alias { None, OutIsA, OutIsB };

std::vector<float> add_cpu(const std::vector<float>& a,
                           const std::vector<float>& b) {
  std::vector<float> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  return out;
}

std::vector<float> run_kernel(const std::vector<float>& a,
                              const std::vector<float>& b, Alias alias) {
  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_out;
  if (alias == Alias::None) d_out.allocate(a.size());

  CUDA_CHECK(cudaMemcpy(d_a.get(), a.data(), d_a.bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), b.data(), d_b.bytes(), cudaMemcpyHostToDevice));

  float* out = d_a.get();
  if (alias == Alias::None) out = d_out.get();
  if (alias == Alias::OutIsB) out = d_b.get();

  CudaStream stream;
  residual_add(out, d_a.get(), d_b.get(), static_cast<int>(a.size()),
               stream.get());
  stream.sync();

  std::vector<float> host(a.size());
  CUDA_CHECK(cudaMemcpy(host.data(), out, host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return host;
}

std::vector<float> gaussian(std::size_t n, float spread, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0f, spread);
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = noise(rng);
  return out;
}

bool run_case(const std::string& name, std::size_t n, Alias alias) {
  // Different spreads so a swapped a and b would still be caught by the
  // aliasing cases, where the two operands play different roles.
  const std::vector<float> a = gaussian(n, 1.0f, 11);
  const std::vector<float> b = gaussian(n, 4.0f, 22);

  // Both sides do one fp32 add of the same pair, so the result is exact.
  return report(name, compare(run_kernel(a, b, alias), add_cpu(a, b),
                              Tolerance::elementwise()));
}

}  // namespace

int main() {
  TestRun run;

  // Prefill residual stream, 8 tokens of d_model 768.
  run.add(run_case("prefill_6144", 8 * 768, Alias::None));
  // Decode, one token.
  run.add(run_case("decode_768", 768, Alias::None));
  // n not a multiple of the block size.
  run.add(run_case("odd_n_1023", 1023, Alias::None));
  // Fewer elements than one block.
  run.add(run_case("tiny_17", 17, Alias::None));
  // The two calls the block forward actually makes.
  run.add(run_case("in_place_out_is_a", 4096, Alias::OutIsA));
  run.add(run_case("in_place_out_is_b", 4096, Alias::OutIsB));

  return run.finish("test_residual");
}
