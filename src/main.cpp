// nano-infer CLI.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/model.hpp"
#include "nanoinfer/weights.hpp"

namespace {

constexpr int kTopK = 5;

void print_usage(const char* argv0) {
  std::printf("usage: %s <checkpoint.bin> [token_id ...]\n", argv0);
  std::printf("  no ids    print the model summary\n");
  std::printf("  with ids  run one prefill pass and print the top %d next tokens\n",
              kTopK);
}

double to_mib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_device_info() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

  std::printf("device      %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);
  std::printf("memory      %.0f MiB free of %.0f MiB\n", to_mib(free_bytes),
              to_mib(total_bytes));
}

void print_model_info(const nanoinfer::ModelWeights& weights,
                      const std::string& path) {
  const nanoinfer::GPT2Config& cfg = weights.config();
  std::printf("\ncheckpoint  %s\n", path.c_str());
  std::printf("dtype       %s\n", nanoinfer::dtype_name(weights.dtype()));
  std::printf("layers      %u\n", cfg.n_layer);
  std::printf("heads       %u (head_dim %u)\n", cfg.n_head, cfg.head_dim());
  std::printf("d_model     %u\n", cfg.n_embd);
  std::printf("context     %u\n", cfg.n_ctx);
  std::printf("vocab       %u\n", cfg.n_vocab);
  std::printf("tensors     %zu\n", weights.tensor_count());
  std::printf("weights     %.1f MiB on device\n", to_mib(weights.device_bytes()));
  std::printf("kv cache    %.1f MiB per sequence at full context\n",
              to_mib(cfg.kv_cache_bytes(weights.dtype())));
}

std::vector<int> parse_ids(int count, char** values, std::uint32_t n_vocab) {
  std::vector<int> ids;
  ids.reserve(static_cast<std::size_t>(count));

  for (int i = 0; i < count; ++i) {
    char* end = nullptr;
    const long value = std::strtol(values[i], &end, 10);
    if (end == values[i] || *end != '\0' || value < 0 ||
        static_cast<unsigned long>(value) >= n_vocab) {
      throw std::runtime_error("token id '" + std::string(values[i]) +
                               "' is outside 0.." + std::to_string(n_vocab - 1));
    }
    ids.push_back(static_cast<int>(value));
  }
  return ids;
}

// Ids and not text. The BPE tokeniser is a later milestone, so the caller
// supplies ids and reads ids back.
void print_top_k(const std::vector<float>& row) {
  std::vector<int> order(row.size());
  std::iota(order.begin(), order.end(), 0);

  const std::size_t count = std::min<std::size_t>(kTopK, row.size());
  std::partial_sort(
      order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count),
      order.end(), [&row](int a, int b) {
        return row[static_cast<std::size_t>(a)] > row[static_cast<std::size_t>(b)];
      });

  std::printf("\ntop %zu next tokens\n", count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t id = static_cast<std::size_t>(order[i]);
    std::printf("  %zu  id %6zu  logit %8.3f\n", i + 1, id,
                static_cast<double>(row[id]));
  }
}

void run_prefill(const nanoinfer::ModelWeights& weights,
                 const std::vector<int>& ids) {
  const nanoinfer::GPT2Config& cfg = weights.config();
  const int tokens = static_cast<int>(ids.size());

  nanoinfer::GPT2Model model(weights, tokens);
  std::printf("workspace   %.1f MiB for %d tokens\n",
              to_mib(model.workspace_bytes()), tokens);

  model.forward(ids.data(), tokens);

  std::vector<float> last(cfg.n_vocab);
  const float* row =
      model.logits() + static_cast<long long>(tokens - 1) * cfg.n_vocab;
  CUDA_CHECK(cudaMemcpy(last.data(), row, last.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  print_top_k(last);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 2;
  }

  try {
    print_device_info();

    const std::string path = argv[1];
    nanoinfer::ModelWeights weights = nanoinfer::ModelWeights::load(path);
    print_model_info(weights, path);

    if (argc > 2) {
      run_prefill(weights, parse_ids(argc - 2, argv + 2, weights.config().n_vocab));
    }
    return 0;
  } catch (const std::exception& err) {
    std::fprintf(stderr, "error: %s\n", err.what());
    return 1;
  }
}
