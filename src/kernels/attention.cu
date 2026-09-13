// Attention, unfused. Three launches with the score matrix materialised between
// them: QKᵀ, a masked row softmax, then the value sum.
//
// Materialising is the whole point of this version. It is the baseline the fused
// kernel gets measured against, and the score matrix it writes is the tensor you
// read when a head misbehaves. Design notes and the mask reasoning live in
// docs/kernels.md.

#include "nanoinfer/kernels/attention.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "nanoinfer/cuda_utils.hpp"
#include "nanoinfer/device_ops.cuh"
#include "nanoinfer/launch.cuh"

namespace nanoinfer {
namespace {

// One warp per key in the score kernel, so the block must be whole warps.
static_assert(kBlockThreads % kWarpSize == 0, "block must be whole warps");

// One thread per output channel in the context kernel.
constexpr int kMaxHeadDim = 1024;

constexpr int kReduceSlots = 32;  // one slot per warp at the 1024-thread maximum

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("nano-infer attention: " + message);
}

__device__ __forceinline__ const float* row_of(const float* base, int row,
                                               int stride) {
  return base + static_cast<long long>(row) * stride;
}

__device__ __forceinline__ long long score_offset(int head, int query,
                                                  int n_query, int n_key) {
  return (static_cast<long long>(head) * n_query + query) * n_key;
}

// Keys a query row is allowed to see, counting from zero.
__device__ __forceinline__ int visible_keys(int query, int pos_offset,
                                            int n_key) {
  const int limit = pos_offset + query + 1;
  return limit < n_key ? limit : n_key;
}

dim3 row_grid(const AttentionShape& shape) {
  return dim3(static_cast<unsigned>(shape.n_query),
              static_cast<unsigned>(shape.n_head));
}

// One block per (query, head), one warp per key. Lanes stride the head
// dimension, so the addresses a warp touches in a key row are contiguous. One
// thread per key instead would have neighbouring threads reading a full row
// apart and turn each 128-byte transaction into 32.
__global__ void scores_kernel(float* __restrict__ scores,
                              const float* __restrict__ q,
                              const float* __restrict__ k, int n_query,
                              int n_key, int head_dim, int q_stride,
                              int kv_stride, int pos_offset, float scale) {
  extern __shared__ float q_row[];

  const int query = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int warps = threads / kWarpSize;

  // Every warp in the block reads the same query row, so it is worth staging
  // once. head_dim is 64 in GPT-2, a quarter of a kilobyte.
  const float* q_src = row_of(q, query, q_stride) + head * head_dim;
  for (int c = tid; c < head_dim; c += threads) q_row[c] = q_src[c];
  __syncthreads();

  float* row = scores + score_offset(head, query, n_query, n_key);
  const int visible = visible_keys(query, pos_offset, n_key);

  // The bound depends on the warp index and never on the lane, so every lane of
  // a warp makes the same number of trips. warp_reduce_sum shuffles under the
  // full mask and reads undefined data from any lane that left early.
  for (int key = warp; key < n_key; key += warps) {
    if (key >= visible) {
      if (lane == 0) row[key] = -INFINITY;
      continue;
    }

    const float* k_row = row_of(k, key, kv_stride) + head * head_dim;
    float dot = 0.0f;
    for (int c = lane; c < head_dim; c += kWarpSize) {
      dot = fmaf(q_row[c], k_row[c], dot);
    }
    dot = warp_reduce_sum(dot);

    // Scaling belongs here and not after the softmax, which is invariant to an
    // additive shift and not to a multiplicative one.
    if (lane == 0) row[key] = dot * scale;
  }
}

// One block per row, two passes over the visible prefix. In place, so no
// __restrict__.
__global__ void softmax_kernel(float* scores, int n_query, int n_key,
                               int pos_offset) {
  __shared__ float reduce[kReduceSlots];

  const int query = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);

  float* row = scores + score_offset(head, query, n_query, n_key);
  const int visible = visible_keys(query, pos_offset, n_key);

  // Threads that never enter the loop still reach both reductions, which
  // contain __syncthreads. They carry the identity in.
  float thread_max = -INFINITY;
  for (int j = tid; j < visible; j += threads) {
    thread_max = fmaxf(thread_max, row[j]);
  }
  const float row_max = block_reduce_max(thread_max, reduce);

  float thread_sum = 0.0f;
  for (int j = tid; j < visible; j += threads) {
    const float weight = expf(row[j] - row_max);
    row[j] = weight;
    thread_sum += weight;
  }
  const float total = block_reduce_sum(thread_sum, reduce);
  const float inv_total = 1.0f / total;

  for (int j = tid; j < visible; j += threads) row[j] *= inv_total;

  // Masked positions leave as exact zeros. The context kernel stops at
  // `visible` and never reads them, but a defined buffer is what makes the
  // matrix comparable against attn.probs and worth dumping.
  for (int j = visible + tid; j < n_key; j += threads) row[j] = 0.0f;
}

// One block per (query, head), one thread per output channel. Neighbouring
// threads read neighbouring channels of the same value row, and the probability
// broadcasts across the block.
__global__ void context_kernel(float* __restrict__ out,
                               const float* __restrict__ probs,
                               const float* __restrict__ v, int n_query,
                               int n_key, int head_dim, int kv_stride,
                               int out_stride, int pos_offset) {
  const int channel = static_cast<int>(threadIdx.x);

  // The block rounds up to whole warps, so the tail carries threads with no
  // channel. No barrier follows, so they leave immediately.
  if (channel >= head_dim) return;

  const int query = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int column = head * head_dim + channel;

  const float* row = probs + score_offset(head, query, n_query, n_key);
  const int visible = visible_keys(query, pos_offset, n_key);

  float acc = 0.0f;
  for (int key = 0; key < visible; ++key) {
    acc = fmaf(row[key], row_of(v, key, kv_stride)[column], acc);
  }

  out[static_cast<long long>(query) * out_stride + column] = acc;
}

}  // namespace

AttentionShape AttentionShape::packed(int tokens, int n_head, int head_dim) {
  const int d = n_head * head_dim;
  AttentionShape shape;
  shape.n_query = tokens;
  shape.n_key = tokens;
  shape.n_head = n_head;
  shape.head_dim = head_dim;
  shape.q_stride = 3 * d;
  shape.kv_stride = 3 * d;
  shape.out_stride = d;
  return shape;
}

void attention_validate(const AttentionShape& shape) {
  if (shape.n_query <= 0 || shape.n_key <= 0) {
    fail("n_query and n_key must both be positive");
  }
  if (shape.n_head <= 0) fail("n_head must be positive");
  if (shape.head_dim <= 0 || shape.head_dim > kMaxHeadDim) {
    fail("head_dim " + std::to_string(shape.head_dim) + " is outside 1.." +
         std::to_string(kMaxHeadDim));
  }
  if (shape.pos_offset < 0) fail("pos_offset must not be negative");

  // The last query sits at pos_offset + n_query - 1 and attends to the key at
  // its own position. Fewer keys than that means either a stale cache length or
  // an offset the caller forgot to advance, and the softmax would divide a
  // masked row by a zero sum.
  if (shape.n_key < shape.pos_offset + shape.n_query) {
    fail("n_key " + std::to_string(shape.n_key) +
         " does not reach the last query position " +
         std::to_string(shape.pos_offset + shape.n_query - 1));
  }

  const int d = shape.model_dim();
  if (shape.q_stride < d || shape.kv_stride < d || shape.out_stride < d) {
    fail("row strides must be at least n_head * head_dim = " +
         std::to_string(d));
  }
}

void attention_scores(float* scores, const float* q, const float* k,
                      const AttentionShape& shape, cudaStream_t stream) {
  attention_validate(shape);

  const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
  const std::size_t shared =
      static_cast<std::size_t>(shape.head_dim) * sizeof(float);

  scores_kernel<<<row_grid(shape), kBlockThreads, shared, stream>>>(
      scores, q, k, shape.n_query, shape.n_key, shape.head_dim, shape.q_stride,
      shape.kv_stride, shape.pos_offset, scale);
  CUDA_CHECK_LAUNCH();
}

void attention_softmax(float* scores, const AttentionShape& shape,
                       cudaStream_t stream) {
  attention_validate(shape);

  softmax_kernel<<<row_grid(shape), row_block_size(shape.n_key), 0, stream>>>(
      scores, shape.n_query, shape.n_key, shape.pos_offset);
  CUDA_CHECK_LAUNCH();
}

void attention_context(float* out, const float* probs, const float* v,
                       const AttentionShape& shape, cudaStream_t stream) {
  attention_validate(shape);

  const int threads = ceil_div(shape.head_dim, kWarpSize) * kWarpSize;
  context_kernel<<<row_grid(shape), threads, 0, stream>>>(
      out, probs, v, shape.n_query, shape.n_key, shape.head_dim,
      shape.kv_stride, shape.out_stride, shape.pos_offset);
  CUDA_CHECK_LAUNCH();
}

void attention_forward(float* out, float* scores, const float* q, const float* k,
                       const float* v, const AttentionShape& shape,
                       cudaStream_t stream) {
  attention_scores(scores, q, k, shape, stream);
  attention_softmax(scores, shape, stream);
  attention_context(out, scores, v, shape, stream);
}

void attention_forward_packed(float* out, float* scores, const float* qkv,
                              int tokens, int n_head, int head_dim,
                              cudaStream_t stream) {
  const AttentionShape shape = AttentionShape::packed(tokens, n_head, head_dim);
  const int d = shape.model_dim();
  attention_forward(out, scores, qkv, qkv + d, qkv + 2 * d, shape, stream);
}

}  // namespace nanoinfer
