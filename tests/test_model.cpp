// nano-infer: end-to-end parity with HuggingFace.
//
// Every kernel passing in isolation and the logits still wrong means the wiring
// is wrong, so this test checks the residual stream at each boundary the dump
// records before it looks at the logits. A failure names the block.
//
// Needs the exported checkpoint as well as the reference dump, and skips when
// either is absent.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/model.hpp"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

#ifndef NANOINFER_WEIGHTS_DIR
#define NANOINFER_WEIGHTS_DIR "weights"
#endif

bool file_exists(const std::string& path) {
  std::FILE* handle = std::fopen(path.c_str(), "rb");
  if (handle == nullptr) return false;
  std::fclose(handle);
  return true;
}

std::string checkpoint_path() {
  const char* from_env = std::getenv("NANOINFER_CHECKPOINT");
  if (from_env != nullptr) return from_env;
  return std::string(NANOINFER_WEIGHTS_DIR) + "/gpt2-124m-f32.bin";
}

// An activation near zero inside a tensor whose scale is 50 carries the error of
// the sums that produced it, not of its own magnitude, so the absolute term
// comes from the reference RMS rather than from a constant. Same argument as the
// GEMM floor one kernel down, and a flat term fails a correct engine on
// whichever activations happen to cancel.
Tolerance scaled(const std::vector<float>& reference, Tolerance base) {
  double sum_sq = 0.0;
  for (float value : reference) sum_sq += static_cast<double>(value) * value;
  const double rms =
      reference.empty() ? 0.0 : std::sqrt(sum_sq / static_cast<double>(reference.size()));
  base.absolute = std::max(base.absolute, static_cast<float>(base.relative * rms));
  return base;
}

bool check_stage(const std::vector<float>& actual, const std::string& name,
                 Tolerance base) {
  const NpyArray reference = load_npy(ref_path(name));
  return check_host("parity_" + name, actual, ref_path(name),
                    scaled(reference.data, base));
}

// The prediction is what a user sees. Logits inside tolerance with an argmax
// that disagrees is still a broken engine, and a top pair separated by less than
// the tolerance is worth printing rather than hiding.
bool check_argmax(const std::vector<float>& actual, const std::string& name) {
  const NpyArray reference = load_npy(ref_path(name));
  const std::size_t rows = reference.rows();
  const std::size_t cols = reference.cols();

  std::size_t predicted = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* got = actual.data() + r * cols;
    const float* want = reference.data.data() + r * cols;
    const std::size_t got_best =
        static_cast<std::size_t>(std::max_element(got, got + cols) - got);
    const std::size_t want_best =
        static_cast<std::size_t>(std::max_element(want, want + cols) - want);

    if (got_best != want_best) {
      std::printf("FAIL %-28s row %zu picks %zu at %.6f, reference picks %zu at %.6f\n",
                  "argmax", r, got_best, static_cast<double>(got[got_best]),
                  want_best, static_cast<double>(want[want_best]));
      return false;
    }
    predicted = got_best;
  }

  std::printf("PASS %-28s %zu rows, last predicts token %zu\n", "argmax", rows,
              predicted);
  return true;
}

struct Stage {
  std::string reference;  // name in the dump
  std::string stage;      // name the model emits
  int blocks = 0;         // transformer blocks this tensor has passed through
};

// h.<i>.in is the stream entering block i, which is the previous block's output
// on our side, or the embedding for block 0. ln_f.in is the last block's output.
// Checking the inputs as well as the outputs costs nothing and narrows a failure
// to one block rather than to the stack above it.
std::vector<Stage> stage_map(int n_layer) {
  std::vector<Stage> stages;
  stages.push_back({"embed.out", "embed.out", 0});
  for (int i = 0; i < n_layer; ++i) {
    const std::string index = std::to_string(i);
    const std::string prior =
        i == 0 ? "embed.out" : "h." + std::to_string(i - 1) + ".out";
    stages.push_back({"h." + index + ".in", prior, i});
    stages.push_back({"h." + index + ".out", "h." + index + ".out", i + 1});
  }
  stages.push_back({"ln_f.in", "h." + std::to_string(n_layer - 1) + ".out", n_layer});
  stages.push_back({"ln_f.out", "ln_f.out", n_layer});
  return stages;
}

int run() {
  if (!reference_available()) {
    std::printf("SKIP no reference data in %s, run: python tools/dump_reference.py\n",
                NANOINFER_TEST_DATA_DIR);
    return kSkipExitCode;
  }

  const std::string checkpoint = checkpoint_path();
  if (!file_exists(checkpoint)) {
    std::printf("SKIP no checkpoint at %s, run: python tools/export_gpt2.py "
                "--model gpt2 --out %s\n",
                checkpoint.c_str(), checkpoint.c_str());
    return kSkipExitCode;
  }

  ModelWeights weights = ModelWeights::load(checkpoint);
  const GPT2Config& cfg = weights.config();

  const NpyArray token_ids = load_npy(ref_path("tokens"));
  const NpyArray reference_logits = load_npy(ref_path("logits"));
  const int tokens = static_cast<int>(token_ids.size());

  // The dump and the checkpoint are produced by two separate scripts. A mismatch
  // here means they came from different models, which would otherwise surface as
  // a wall of tolerance failures.
  if (reference_logits.cols() != cfg.n_vocab ||
      reference_logits.rows() != static_cast<std::size_t>(tokens)) {
    std::printf("FAIL %-28s dump is %s for %d tokens, checkpoint has vocab %u\n",
                "dump_matches_checkpoint", reference_logits.shape_string().c_str(),
                tokens, cfg.n_vocab);
    return 1;
  }

  std::vector<int> ids(static_cast<std::size_t>(tokens));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<int>(std::lround(token_ids.data[i]));
  }

  GPT2Model model(weights, tokens);

  std::map<std::string, std::vector<float>> captured;
  model.forward(ids.data(), tokens,
                [&captured](const std::string& stage, const float* device, int rows,
                            int cols) {
                  std::vector<float> host(static_cast<std::size_t>(rows) *
                                          static_cast<std::size_t>(cols));
                  CUDA_CHECK(cudaMemcpy(host.data(), device,
                                        host.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost));
                  captured.emplace(stage, std::move(host));
                });

  std::printf("checkpoint  %s\nworkspace   %.1f MiB for %d tokens\n\n",
              checkpoint.c_str(),
              static_cast<double>(model.workspace_bytes()) / (1024.0 * 1024.0),
              tokens);

  TestRun run;
  int compared = 0;

  for (const Stage& entry : stage_map(static_cast<int>(cfg.n_layer))) {
    // Only the dumped layers are on disk. --layers all fills in the rest.
    if (!file_exists(ref_path(entry.reference))) continue;

    const auto found = captured.find(entry.stage);
    if (found == captured.end()) {
      std::printf("FAIL %-28s model emitted no stage %s\n", entry.reference.c_str(),
                  entry.stage.c_str());
      run.add(false);
      continue;
    }

    // The embedding is one fp32 add of two table rows, so it should land on the
    // reference exactly. Everything downstream carries a GEMM chain.
    Tolerance base = entry.reference == "embed.out" ? Tolerance::elementwise()
                                                    : Tolerance::gemm();

    // Both bases size one kernel. A tensor twelve blocks down the residual
    // stream carries the rounding of every GEMM and softmax above it, and each
    // block feeds its error to the next, so the bar has to widen with depth.
    // Tolerance::logits() already concedes this, sitting ten times looser than
    // Tolerance::gemm() because logits leave the full stack. Scaling linearly
    // with block count makes the concession explicit and monotone: block 0 keeps
    // the single-kernel figure, and the last block lands next to the logits
    // figure instead of facing a cliff one LayerNorm before it.
    base.relative *= static_cast<float>(1 + entry.blocks);

    run.add(check_stage(found->second, entry.reference, base));
    ++compared;
  }

  if (compared == 0) {
    std::printf("FAIL %-28s no intermediate stage was on disk\n", "stage_coverage");
    run.add(false);
  }

  const std::vector<float>& logits = captured.at("logits");
  run.add(check_stage(logits, "logits", Tolerance::logits()));
  run.add(check_argmax(logits, "logits"));

  return run.finish("test_model");
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& err) {
    std::fprintf(stderr, "FAIL test_model: %s\n", err.what());
    return 1;
  }
}
