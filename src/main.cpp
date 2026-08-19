// nano-infer CLI.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "nanoinfer/config.hpp"
#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/weights.hpp"

namespace {

void print_usage(const char* argv0) {
  std::printf("usage: %s <checkpoint.bin>\n", argv0);
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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    print_usage(argv[0]);
    return 2;
  }

  try {
    print_device_info();

    const std::string path = argv[1];
    nanoinfer::ModelWeights model = nanoinfer::ModelWeights::load(path);

    const nanoinfer::GPT2Config& cfg = model.config();
    std::printf("\ncheckpoint  %s\n", path.c_str());
    std::printf("dtype       %s\n", nanoinfer::dtype_name(model.dtype()));
    std::printf("layers      %u\n", cfg.n_layer);
    std::printf("heads       %u (head_dim %u)\n", cfg.n_head, cfg.head_dim());
    std::printf("d_model     %u\n", cfg.n_embd);
    std::printf("context     %u\n", cfg.n_ctx);
    std::printf("vocab       %u\n", cfg.n_vocab);
    std::printf("tensors     %zu\n", model.tensor_count());
    std::printf("weights     %.1f MiB on device\n", to_mib(model.device_bytes()));
    std::printf("kv cache    %.1f MiB per sequence at full context\n",
                to_mib(cfg.kv_cache_bytes(model.dtype())));

    return 0;
  } catch (const std::exception& err) {
    std::fprintf(stderr, "error: %s\n", err.what());
    return 1;
  }
}
