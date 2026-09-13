// nano-infer: attention tests.
//
// Synthetic cases check the three stages against a double-precision oracle and
// assert the causal structure separately. A mask off by one is numerically tiny
// and semantically fatal, so it gets a check that fails on the structure rather
// than on a tolerance.
//
// The parity case needs no extra dump flags. h.0.attn.qkv.out is the packed QKV
// this kernel consumes, h.0.attn.probs is the probability matrix it produces,
// and h.0.attn.proj.in is the merged head output, which is the attention result
// by construction.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/kernels/attention.cuh"
#include "reference.hpp"

using namespace nanoinfer;
using namespace nanoinfer::test;

namespace {

// One upload backs Q, K and V. Packed cases point all three at the same buffer
// with different offsets, which is how the QKV projection writes them, and the
// decode case moves the Q offset to the last row without moving K or V.
struct Layout {
  std::vector<float> data;
  int q_offset = 0;
  int k_offset = 0;
  int v_offset = 0;
  AttentionShape shape;
};

struct Activations {
  std::vector<float> probs;
  std::vector<float> out;
};

std::size_t prob_count(const AttentionShape& s) {
  return static_cast<std::size_t>(s.score_elements());
}

std::size_t out_count(const AttentionShape& s) {
  return static_cast<std::size_t>(s.n_query) *
         static_cast<std::size_t>(s.out_stride);
}

Layout packed_layout(int tokens, int n_head, int head_dim) {
  Layout layout;
  layout.shape = AttentionShape::packed(tokens, n_head, head_dim);
  const int d = layout.shape.model_dim();
  layout.k_offset = d;
  layout.v_offset = 2 * d;
  layout.data.assign(static_cast<std::size_t>(tokens) * 3 * d, 0.0f);
  return layout;
}

Layout random_packed(int tokens, int n_head, int head_dim, unsigned seed) {
  Layout layout = packed_layout(tokens, n_head, head_dim);
  std::mt19937 rng(seed);  // fixed, a flaky numerical test teaches you to ignore it
  std::normal_distribution<float> noise(0.0f, 1.0f);
  for (float& value : layout.data) value = noise(rng);
  return layout;
}

// Decode: one query row at the end of a filled cache. K and V still span every
// row, and pos_offset carries the absolute position the mask needs.
Layout decode_from(const Layout& prefill) {
  Layout layout = prefill;
  layout.shape.n_query = 1;
  layout.shape.pos_offset = prefill.shape.n_key - 1;
  layout.q_offset =
      prefill.q_offset + (prefill.shape.n_key - 1) * prefill.shape.q_stride;
  return layout;
}

// Scores of exactly 96 + 1.5 * key, past the fp32 exp overflow point at
// ln(FLT_MAX) = 88.72. Every constant is a dyadic rational, so the dot product
// is bit-identical in fp32 and in the double oracle whatever order it sums in,
// and the only thing left under test is the max subtraction. Without it the
// whole case comes back NaN.
Layout overflow_layout(int tokens, int n_head) {
  constexpr int kHeadDim = 64;  // scale is 1/8, which the constants below assume
  Layout layout = packed_layout(tokens, n_head, kHeadDim);

  for (int t = 0; t < tokens; ++t) {
    float* row = layout.data.data() +
                 static_cast<std::size_t>(t) * layout.shape.q_stride;
    const float key_value = 3.0f + 3.0f * static_cast<float>(t) / 64.0f;
    for (int h = 0; h < n_head; ++h) {
      for (int c = 0; c < kHeadDim; ++c) {
        const int column = h * kHeadDim + c;
        row[layout.q_offset + column] = 4.0f;
        row[layout.k_offset + column] = key_value;
        row[layout.v_offset + column] =
            0.5f * static_cast<float>(t + 1) + 0.25f * static_cast<float>(c);
      }
    }
  }
  return layout;
}

// Oracle in double precision. The kernel runs in fp32, so the reference has to
// carry more precision than the thing it judges.
Activations attention_cpu(const Layout& layout) {
  const AttentionShape& s = layout.shape;
  const float* base = layout.data.data();

  Activations ref;
  ref.probs.assign(prob_count(s), 0.0f);
  ref.out.assign(out_count(s), 0.0f);

  const double scale = 1.0 / std::sqrt(static_cast<double>(s.head_dim));
  std::vector<double> weights(static_cast<std::size_t>(s.n_key), 0.0);

  for (int h = 0; h < s.n_head; ++h) {
    const int column = h * s.head_dim;
    for (int i = 0; i < s.n_query; ++i) {
      const int visible = std::min(s.n_key, s.pos_offset + i + 1);
      const float* q_row = base + layout.q_offset +
                           static_cast<std::size_t>(i) * s.q_stride + column;

      double largest = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < visible; ++j) {
        const float* k_row = base + layout.k_offset +
                             static_cast<std::size_t>(j) * s.kv_stride + column;
        double dot = 0.0;
        for (int c = 0; c < s.head_dim; ++c) {
          dot += static_cast<double>(q_row[c]) * static_cast<double>(k_row[c]);
        }
        weights[static_cast<std::size_t>(j)] = dot * scale;
        largest = std::max(largest, weights[static_cast<std::size_t>(j)]);
      }

      double total = 0.0;
      for (int j = 0; j < visible; ++j) {
        const std::size_t w = static_cast<std::size_t>(j);
        weights[w] = std::exp(weights[w] - largest);
        total += weights[w];
      }

      float* probs =
          ref.probs.data() +
          (static_cast<std::size_t>(h) * s.n_query + i) * s.n_key;
      for (int j = 0; j < visible; ++j) {
        probs[j] = static_cast<float>(weights[static_cast<std::size_t>(j)] / total);
      }

      float* out = ref.out.data() +
                   static_cast<std::size_t>(i) * s.out_stride + column;
      for (int c = 0; c < s.head_dim; ++c) {
        double acc = 0.0;
        for (int j = 0; j < visible; ++j) {
          const float* v_row = base + layout.v_offset +
                               static_cast<std::size_t>(j) * s.kv_stride + column;
          acc += (weights[static_cast<std::size_t>(j)] / total) *
                 static_cast<double>(v_row[c]);
        }
        out[c] = static_cast<float>(acc);
      }
    }
  }
  return ref;
}

// Runs the stages one at a time rather than attention_forward, so a failure
// names a kernel. The parity case below covers the composed entry point.
Activations attention_gpu(const Layout& layout) {
  const AttentionShape& s = layout.shape;

  DeviceBuffer<float> d_data(layout.data.size());
  DeviceBuffer<float> d_probs(prob_count(s));
  DeviceBuffer<float> d_out(out_count(s));

  CUDA_CHECK(cudaMemcpy(d_data.get(), layout.data.data(), d_data.bytes(),
                        cudaMemcpyHostToDevice));

  // Poison, so a position no thread writes fails instead of reading as zero.
  CUDA_CHECK(cudaMemset(d_probs.get(), 0x7f, d_probs.bytes()));
  CUDA_CHECK(cudaMemset(d_out.get(), 0x7f, d_out.bytes()));

  CudaStream stream;
  attention_scores(d_probs.get(), d_data.get() + layout.q_offset,
                   d_data.get() + layout.k_offset, s, stream.get());
  attention_softmax(d_probs.get(), s, stream.get());
  attention_context(d_out.get(), d_probs.get(), d_data.get() + layout.v_offset, s,
                    stream.get());
  stream.sync();

  Activations result;
  result.probs.resize(prob_count(s));
  result.out.resize(out_count(s));
  CUDA_CHECK(cudaMemcpy(result.probs.data(), d_probs.get(), d_probs.bytes(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.out.data(), d_out.get(), d_out.bytes(),
                        cudaMemcpyDeviceToHost));
  return result;
}

// The two properties the oracle comparison states least clearly. A mask that
// admits one future key shifts a row by a fraction of a percent, which reads as
// accumulated error rather than as a bug.
bool check_structure(const std::string& label, const std::vector<float>& probs,
                     const AttentionShape& s) {
  for (int h = 0; h < s.n_head; ++h) {
    for (int i = 0; i < s.n_query; ++i) {
      const float* row =
          probs.data() + (static_cast<std::size_t>(h) * s.n_query + i) * s.n_key;
      const int visible = std::min(s.n_key, s.pos_offset + i + 1);

      double total = 0.0;
      for (int j = 0; j < s.n_key; ++j) {
        if (j >= visible && row[j] != 0.0f) {
          std::printf("FAIL %-28s head %d row %d leaks key %d, weight %.3e\n",
                      label.c_str(), h, i, j, static_cast<double>(row[j]));
          return false;
        }
        total += static_cast<double>(row[j]);
      }

      if (std::fabs(total - 1.0) > 1e-5) {
        std::printf("FAIL %-28s head %d row %d sums to %.9g\n", label.c_str(), h,
                    i, total);
        return false;
      }
    }
  }

  std::printf("PASS %-28s mask and row sums\n", label.c_str());
  return true;
}

bool run_case(const std::string& name, const Layout& layout) {
  const AttentionShape& s = layout.shape;
  const Activations expected = attention_cpu(layout);
  const Activations actual = attention_gpu(layout);

  bool ok = report(name + ".probs",
                   compare(actual.probs, expected.probs, Tolerance::attention()),
                   static_cast<std::size_t>(s.n_key));
  ok &= report(name + ".out",
               compare(actual.out, expected.out, Tolerance::attention()),
               static_cast<std::size_t>(s.out_stride));
  ok &= check_structure(name + ".mask", actual.probs, s);
  return ok;
}

// Layer 0 attention end to end, from the real QKV projection output to the
// merged head output c_proj consumes. Uses the composed launcher.
bool run_parity() {
  const NpyArray qkv = load_npy(ref_path("h.0.attn.qkv.out"));
  const NpyArray probs = load_npy(ref_path("h.0.attn.probs"));

  if (probs.shape.size() != 4) {
    std::printf("FAIL %-28s probs is %s, expected [batch, head, query, key]\n",
                "hf_parity_h0_attn", probs.shape_string().c_str());
    return false;
  }

  const int tokens = static_cast<int>(qkv.rows());
  const int n_head = static_cast<int>(probs.shape[1]);
  const int d = static_cast<int>(qkv.cols()) / 3;

  if (n_head <= 0 || d <= 0 || d % n_head != 0) {
    std::printf("FAIL %-28s %d heads do not divide d_model %d\n",
                "hf_parity_h0_attn", n_head, d);
    return false;
  }
  const int head_dim = d / n_head;

  DeviceBuffer<float> d_qkv(qkv.data.size());
  DeviceBuffer<float> d_probs(static_cast<std::size_t>(n_head) * tokens * tokens);
  DeviceBuffer<float> d_out(static_cast<std::size_t>(tokens) * d);

  CUDA_CHECK(cudaMemcpy(d_qkv.get(), qkv.data.data(), d_qkv.bytes(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_probs.get(), 0x7f, d_probs.bytes()));
  CUDA_CHECK(cudaMemset(d_out.get(), 0x7f, d_out.bytes()));

  CudaStream stream;
  attention_forward_packed(d_out.get(), d_probs.get(), d_qkv.get(), tokens,
                           n_head, head_dim, stream.get());
  stream.sync();

  std::vector<float> host_probs(d_probs.count());
  std::vector<float> host_out(d_out.count());
  CUDA_CHECK(cudaMemcpy(host_probs.data(), d_probs.get(), d_probs.bytes(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_out.data(), d_out.get(), d_out.bytes(),
                        cudaMemcpyDeviceToHost));

  bool ok = check_host("hf_parity_h0_attn.probs", host_probs,
                       ref_path("h.0.attn.probs"), Tolerance::attention());
  ok &= check_host("hf_parity_h0_attn.out", host_out,
                   ref_path("h.0.attn.proj.in"), Tolerance::attention());
  return ok;
}

// A shape the kernels cannot serve is a caller bug, and the most likely one is
// a decode step that advances the cache without advancing the offset. It has to
// be caught before a launch, not after a NaN.
bool run_rejects() {
  const int tokens = 4;
  const int n_head = 2;
  const int head_dim = 32;

  struct Bad {
    const char* what;
    AttentionShape shape;
  };

  AttentionShape stale_offset = AttentionShape::packed(tokens, n_head, head_dim);
  stale_offset.n_query = 1;
  stale_offset.pos_offset = tokens;  // one past the last cached key

  AttentionShape short_stride = AttentionShape::packed(tokens, n_head, head_dim);
  short_stride.out_stride = n_head * head_dim - 1;

  AttentionShape negative_offset = AttentionShape::packed(tokens, n_head, head_dim);
  negative_offset.pos_offset = -1;

  AttentionShape empty = AttentionShape::packed(0, n_head, head_dim);

  const Bad cases[] = {{"pos_offset past the cache", stale_offset},
                       {"out_stride below d_model", short_stride},
                       {"negative pos_offset", negative_offset},
                       {"no tokens", empty}};

  for (const Bad& bad : cases) {
    try {
      attention_validate(bad.shape);
    } catch (const std::runtime_error&) {
      continue;
    }
    std::printf("FAIL %-28s accepted: %s\n", "rejects_bad_shapes", bad.what);
    return false;
  }

  std::printf("PASS %-28s %zu shapes rejected\n", "rejects_bad_shapes",
              sizeof(cases) / sizeof(cases[0]));
  return true;
}

}  // namespace

int main() {
  TestRun run;

  // Prefill at the real head width, several heads, triangular mask.
  run.add(run_case("prefill_4h_8x64", random_packed(8, 4, 64, 1234u)));
  // n_key past the block size, so both the key loop and the softmax stride.
  run.add(run_case("long_rows_2h_300x32", random_packed(300, 2, 32, 2345u)));
  // head_dim off a warp multiple. Lanes past the tail contribute nothing to the
  // dot product and the context block still rounds up to whole warps.
  run.add(run_case("ragged_head_3h_5x20", random_packed(5, 3, 20, 3456u)));
  // Fewer keys than warps in the score block, so most warps take no trip.
  run.add(run_case("narrow_1h_3x64", random_packed(3, 1, 64, 4567u)));
  // One token attending to itself. The softmax denominator is a single term.
  run.add(run_case("single_token_2h_1x64", random_packed(1, 2, 64, 5678u)));
  // Decode. One query row, a full cache behind it, and the mask driven entirely
  // by pos_offset.
  run.add(run_case("decode_2h_1x16", decode_from(random_packed(16, 2, 32, 6789u))));
  // The overflow trap. See overflow_layout.
  run.add(run_case("overflow_2h_6x64", overflow_layout(6, 2)));

  run.add(run_rejects());

  if (reference_available()) {
    run.add(run_parity());
  } else {
    std::printf("SKIP hf_parity_h0_attn, run: python tools/dump_reference.py\n");
  }

  return run.finish("test_attention");
}
