// nano-infer: launch configuration shared by the kernel launchers.
//
// Every launcher needs the same two decisions, how many threads per block and
// how many blocks. Three kernels growing three private copies of the rule is how
// a tuning change ends up half applied, so the rule lives here.
//
// The file is .cuh rather than .hpp because it reuses kWarpSize and ceil_div
// from device_ops.cuh. Defining a second warp constant under a second name is
// worse than the naming bend. Only .cu translation units include it.
#pragma once

#include "nanoinfer/device_ops.cuh"

namespace nanoinfer {

// 256 divides 768 and 3072 evenly and leaves eight warps per block to hide
// memory latency. 1024 idles threads on a 768-wide row, 128 halves the warps
// available.
constexpr int kBlockThreads = 256;

// Ceiling on the blocks a flat elementwise launch asks for. Every such kernel
// runs a grid-stride loop, so this changes occupancy and never changes the
// result. 4096 blocks of 256 threads is roughly four full waves on a 132-SM
// card, past the point where more blocks buy anything and short enough that the
// stride tail stays one extra iteration.
constexpr int kMaxGridBlocks = 4096;

// Block tile for the naive GEMM. One warp wide, so the 32 lanes cover 128
// consecutive bytes of a B row and the load retires as one transaction. Eight
// rows stack eight warps to reach kBlockThreads.
constexpr int kGemmTileN = kWarpSize;                     // 32
constexpr int kGemmTileM = kBlockThreads / kGemmTileN;    // 8

// Threads for a kernel that gives one block to a row and strides across it.
// Rows narrower than a block round up to a warp multiple, since a partial warp
// wastes lanes inside every shuffle.
inline int row_block_size(int cols) {
  if (cols <= 0) return kWarpSize;
  if (cols >= kBlockThreads) return kBlockThreads;
  return ceil_div(cols, kWarpSize) * kWarpSize;
}

// Blocks for a flat grid-stride launch over `count` elements. Takes long long
// so a caller holding a size_t element count cannot silently wrap on the way
// in.
inline int elementwise_grid(long long count, int block = kBlockThreads) {
  if (count <= 0 || block <= 0) return 0;
  const long long blocks = (count + block - 1) / block;
  return static_cast<int>(blocks < kMaxGridBlocks ? blocks : kMaxGridBlocks);
}

}  // namespace nanoinfer
