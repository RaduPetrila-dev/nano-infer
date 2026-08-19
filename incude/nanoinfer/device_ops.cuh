// nano-infer: device-side building blocks.
//
// Header-only and __forceinline__ on purpose. A __device__ function defined in
// one .cu and called from another needs CUDA_SEPARABLE_COMPILATION, and device
// linking blocks cross-file inlining. These functions sit in the innermost loop
// of every kernel in the engine, so a real call here would cost more than the
// work being done.
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
// Butterfly pattern with __shfl_xor_sync, not the tree with __shfl_down_sync.
// Both take log2(32) = 5 shuffles. The difference is where the answer ends up:
// down leaves it in lane 0 only, xor leaves it in all 32 lanes. LayerNorm,
// softmax and attention all need every lane to hold the reduced value, so xor
// removes a sixth broadcast shuffle from each reduction.
//
// kFullMask assumes all 32 lanes are active. Never call these under a branch
// that some lanes skip. Diverged lanes make the shuffle read undefined data,
// and the failure is silent.

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

// Sum and sum-of-squares in one pass. Two independent shuffle chains issue back
// to back and hide each other's latency, which a caller running two separate
// warp_reduce_sum calls also gets. The reason to fuse is register pressure at
// the call site, not instruction count.
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
// Two stages. Every warp reduces internally, one lane per warp writes to shared
// memory, then warp 0 reduces those partials. Shared memory holds 32 floats,
// which covers the 1024-thread maximum block size.
//
// The caller supplies the shared array so a kernel can reuse one allocation for
// several reductions instead of paying for a new one each time.

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = ceil_div(static_cast<int>(blockDim.x), kWarpSize);

  value = warp_reduce_sum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();

  // Lanes past the warp count must read an identity, not stale shared memory.
  value = (lane < warps) ? shared[lane] : 0.0f;
  if (warp == 0) value = warp_reduce_sum(value);

  // Warp 0 holds the answer. Broadcast through shared memory so every thread in
  // the block sees it.
  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  return shared[0];
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
  return shared[0];
}

// ---------------------------------------------------------------------------
// Vectorised access
// ---------------------------------------------------------------------------
//
// A float4 load is one 128-bit instruction instead of four 32-bit ones. With
// d_model = 768, a warp doing float4 loads covers 128 floats per instruction
// and each 32-lane request maps onto four 128-byte cache lines with no waste.
// Scalar loads need four times the instructions for the same bytes, and on a
// bandwidth-bound kernel like LayerNorm the issue rate becomes the ceiling.
//
// The hardware requires 16-byte alignment for a 128-bit access. Checkpoint
// tensors are 256-byte aligned by the loader, so any base pointer is safe, but
// an offset into a row is only safe when the offset is a multiple of 4 floats.
// 768 and 3072 both divide by 4. A head dimension of 64 does too. Check before
// assuming it on a new shape.

__device__ __forceinline__ bool is_vec4_aligned(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0xf) == 0;
}

__device__ __forceinline__ float4 load_vec4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

// __ldg routes through the read-only data cache. Weights are read by every
// block and never written during a forward pass, which is exactly the access
// pattern that path is built for.
__device__ __forceinline__ float4 load_vec4_ro(const float* ptr) {
  return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ __forceinline__ void store_vec4(float* ptr, float4 value) {
  *reinterpret_cast<float4*>(ptr) = value;
}

__device__ __forceinline__ float sum4(float4 v) {
  return v.x + v.y + v.z + v.w;
}

__device__ __forceinline__ float dot4(float4 a, float4 b) {
  // fmaf keeps the chain in one rounding step per multiply-add. Writing a.x*b.x
  // + ... lets the compiler contract into FMA anyway, but only when contraction
  // is enabled. Being explicit makes the numerics independent of build flags.
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
