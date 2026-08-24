// Device-side building blocks: warp and block reductions, vectorised access.
//
// Header-only and __forceinline__ because a __device__ function called across
// translation units needs separable compilation, and device linking blocks
// cross-file inlining. These sit in the innermost loop of every kernel.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace nanoinfer {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

__host__ __device__ constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Warp reductions
// ---------------------------------------------------------------------------
//
// Butterfly with __shfl_xor_sync rather than the tree with __shfl_down_sync.
// Both cost five shuffles, but xor leaves the result in all 32 lanes instead of
// lane 0, which saves a broadcast in LayerNorm, softmax and attention.
//
// kFullMask assumes all 32 lanes are active. Called under a branch some lanes
// skip, the shuffle reads undefined data and fails silently.

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
  }
  return value;
}

// Sum and sum-of-squares together. Fused to cut register pressure at the call
// site, not to save instructions.
__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    a += __shfl_xor_sync(kFullMask, a, offset);
    b += __shfl_xor_sync(kFullMask, b, offset);
  }
}

// ---------------------------------------------------------------------------
// Block reductions
// ---------------------------------------------------------------------------
//
// Each warp reduces internally, leaders write to shared, warp 0 reduces the
// partials. The caller supplies the array so one allocation serves several
// reductions.

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = ceil_div(static_cast<int>(blockDim.x), kWarpSize);

  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  // Lanes past the warp count read an identity, not stale shared memory.
  value = (lane < warps) ? shared[lane] : 0.0f;
  if (warp == 0) value = warp_reduce_sum(value);

  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  const float result = shared[0];

  // Second barrier so the array is reusable. Without it a fast warp reaches the
  // next reduction and overwrites shared[0] while a slow warp is still reading.
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_reduce_max(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = ceil_div(static_cast<int>(blockDim.x), kWarpSize);

  value = warp_reduce_max(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  value = (lane < warps) ? shared[lane] : -FLT_MAX;
  if (warp == 0) value = warp_reduce_max(value);

  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  const float result = shared[0];
  __syncthreads();  // see block_reduce_sum
  return result;
}

// ---------------------------------------------------------------------------
// Vectorised access
// ---------------------------------------------------------------------------
//
// One 128-bit load instead of four 32-bit ones. On a bandwidth-bound kernel the
// instruction issue rate is the ceiling, so this is close to a 4x.
//
// Requires 16-byte alignment. Checkpoint tensors are 256-byte aligned, so base
// pointers are safe. A row offset is safe only when it divides by 4, which 768,
// 3072 and head_dim 64 all do. Check before assuming it on a new shape.

__device__ __forceinline__ bool is_vec4_aligned(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0xf) == 0;
}

__device__ __forceinline__ float4 load_vec4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

// Read-only data cache. Weights are read by every block and never written during
// a forward pass.
__device__ __forceinline__ float4 load_vec4_ro(const float* ptr) {
  return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ void store_vec4(float* ptr, float4 value) {
  *reinterpret_cast<float4*>(ptr) = value;
}

__device__ __forceinline__ float sum4(float4 v) {
  return v.x + v.y + v.z + v.w;
}

// Explicit fmaf so the numerics do not depend on whether contraction is enabled.
__device__ __forceinline__ float dot4(float4 a, float4 b) {
  float acc = fmaf(a.x, b.x, 0.0f);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  acc = fmaf(a.w, b.w, acc);
  return acc;
}

__device__ __forceinline__ float4 scale4(float4 v, float s) {
  return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

}  // namespace nanoinfer
