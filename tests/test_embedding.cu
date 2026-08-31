// nano-infer: embedding tests.
//
// The kernel is a gather, so the cases are about indexing rather than
// arithmetic: repeated ids, the first and last row of the table, a row narrower
// than the block, and a non-zero position offset, which is what decode passes
// once the KV cache holds earlier tokens.
//
// Table entries encode their own coordinates, so a swapped row and column or an
// off-by-one row shows up as a large error rather than a subtle one.

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/kernels/embedding.cuh"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

constexpr int kVocab = 64;
constexpr int kCtx = 32;

// Distinct per (row, col) and far enough apart that a wrong row is obvious.
float table_value(int row, int col, float tag) {
  return tag + static_cast<float>(row) + 0.001f * static_cast<float>(col);
}

std::vector<float> make_table(int rows, int cols, float tag) {
  std::vector<float> out(static_cast<std::size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(r) * cols + c] = table_value(r, c, tag);
    }
  }
  return out;
}

std::vector<float> embedding_cpu(const std::vector<int>& ids,
                                 const std::vector<float>& wte,
                                 const std::vector<float>& wpe, int d_model,
                                 int pos_offset) {
  std::vector<float> out(ids.size() * static_cast<std::size_t>(d_model));
  for (std::size_t t = 0; t < ids.size(); ++t) {
    const std::size_t wte_row = static_cast<std::size_t>(ids[t]) * d_model;
    const std::size_t wpe_row =
        (static_cast<std::size_t>(pos_offset) + t) * d_model;
    for (int c = 0; c < d_model; ++c) {
      out[t * d_model + c] = wte[wte_row + c] + wpe[wpe_row + c];
    }
  }
  return out;
}

std::vector<float> run_kernel(const std::vector<int>& ids,
                              const std::vector<float>& wte,
                              const std::vector<float>& wpe, int d_model,
                              int pos_offset, int n_ctx) {
  DeviceBuffer<int> d_ids(ids.size());
  DeviceBuffer<float> d_wte(wte.size());
  DeviceBuffer<float> d_wpe(wpe.size());
  DeviceBuffer<float> d_out(ids.size() * static_cast<std::size_t>(d_model));

  CUDA_CHECK(cudaMemcpy(d_ids.get(), ids.data(), d_ids.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wte.get(), wte.data(), d_wte.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wpe.get(), wpe.data(), d_wpe.bytes(),
                        cudaMemcpyHostToDevice));

  CudaStream stream;
  embedding_forward(d_out.get(), d_ids.get(), d_wte.get(), d_wpe.get(),
                    static_cast<int>(ids.size()), d_model, pos_offset, kVocab,
                    n_ctx, stream.get());
  stream.sync();

  std::vector<float> out(d_out.count());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out.get(), d_out.bytes(),
                        cudaMemcpyDeviceToHost));
  return out;
}

struct Case {
  std::string name;
  std::vector<int> ids;
  int d_model;
  int pos_offset;
};

bool run_case(const Case& c) {
  const std::vector<float> wte = make_table(kVocab, c.d_model, 1000.0f);
  const std::vector<float> wpe = make_table(kCtx, c.d_model, 2000.0f);

  const std::vector<float> expected =
      embedding_cpu(c.ids, wte, wpe, c.d_model, c.pos_offset);
  const std::vector<float> actual =
      run_kernel(c.ids, wte, wpe, c.d_model, c.pos_offset, kCtx);

  // A gather copies bits, so the only rounding a correct kernel introduces is
  // the single add.
  return report(c.name, compare(actual, expected, Tolerance::elementwise()),
                static_cast<std::size_t>(c.d_model));
}

// A position past n_ctx reads outside wpe and returns plausible garbage, so the
// launcher rejects it rather than producing a wrong logit.
bool run_range_check() {
  const std::vector<int> ids{0, 1};
  const std::vector<float> wte = make_table(kVocab, 8, 1000.0f);
  const std::vector<float> wpe = make_table(kCtx, 8, 2000.0f);

  try {
    run_kernel(ids, wte, wpe, 8, kCtx - 1, kCtx);
  } catch (const std::runtime_error&) {
    std::printf("PASS %-28s position past n_ctx rejected\n", "context_overflow");
    return true;
  }
  std::printf("FAIL %-28s position %d accepted with n_ctx %d\n",
              "context_overflow", kCtx, kCtx);
  return false;
}

}  // namespace

int main() {
  TestRun run;

  const Case cases[] = {
      // Repeated ids plus the first and last row of the table.
      {"gather_5x64", {0, kVocab - 1, 7, 7, 1}, 64, 0},
      // Non-zero offset, the decode path once the cache is warm.
      {"pos_offset_4x64", {3, 9, 9, 12}, 64, 8},
      // Single token at a late position. One block on the whole device.
      {"decode_1x768", {17}, 768, kCtx - 1},
      // Row narrower than one warp. Most of the block sits idle, which is safe
      // here because the kernel contains no barrier.
      {"narrow_3x17", {2, 5, 2}, 17, 0},
      // Row wider than the block, so every thread walks the strided loop.
      {"wide_2x1024", {4, 4}, 1024, 1},
  };

  for (const Case& c : cases) run.add(run_case(c));
  run.add(run_range_check());

  return run.finish("test_embedding");
}
