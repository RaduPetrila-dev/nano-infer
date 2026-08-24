# Kernel design notes

Reasoning that would otherwise bloat the source. One section per kernel: the
maths, the numerical traps, the block configuration, and what each optimisation
measured.

## LayerNorm

```
mean = (1/N) Σ x_c
var  = (1/N) Σ (x_c - mean)²
y_c  = (x_c - mean) / sqrt(var + eps) * gamma_c + beta_c
```

Normalisation runs over the last axis, N = d_model = 768. Variance is the biased
estimator, dividing by N and not N-1, matching `torch.nn.LayerNorm`. Using N-1
at N=768 gives a 0.07% error that fails parity and looks like nothing.

Epsilon is 1e-5, from `GPT2Config.layer_norm_epsilon`, and goes inside the
square root.

### One block per row

The reduction spans the last axis, so every element of a row must reach one
reduction. Threads inside a block share memory and can synchronise. Threads
across blocks cannot without a second launch or a grid sync. One row per block
keeps the reduction local.

### Grid-stride and the identity contribution

`for (c = threadIdx.x; c < cols; c += blockDim.x)` handles any block size and
any column count. The trap is the thread whose loop body never executes. It
still has to call `block_reduce_sum`, because that function contains
`__syncthreads()`, and a barrier some threads never reach hangs the block
permanently. Initialising the accumulator to zero outside the loop handles it:
no elements means contributing the identity.

`tests/test_layernorm.cu` covers this with `ragged_3x100` and `narrow_4x17`.

### Why not E[x²] - E[x]²

The one-pass formula is a single reduction instead of two and is wrong for this
model. You subtract two large nearly equal numbers, the leading digits cancel,
and the answer is left carried by trailing-bit noise.

GPT-2 is exactly that case. The residual stream develops outlier dimensions that
grow with depth and reach magnitudes in the thousands by the later blocks, so
`E[x²]` sits near 1e6 while the variance is orders of magnitude smaller. fp32
carries about seven significant digits and the subtraction discards most of them.

The failure pattern is what makes this worth knowing. A one-pass kernel passes
the layer 0 test cleanly, because early activations are small and well behaved.
It fails at layer 11, by which point three more kernels sit on top of it and
nothing points at the one that has been green all week.

Measured against a double-precision oracle at mean 4000 and spread 1:

| Implementation | max absolute error |
| --- | --- |
| Two-pass | 2.9e-4 |
| E[x²] - E[x]² | 2.2e+00, plus negative variances producing NaN |

Welford's online algorithm is the numerically stable single-pass option. It
needs a parallel merge formula and more registers per thread. Not worth it here,
because the register-cached two-pass already reads global memory once.

### Test tolerances

The synthetic cases compare against a double-precision CPU oracle. The tolerance
for the large-magnitude cases is derived, not guessed. Summing N values of
magnitude `|centre|` in fp32 leaves the sum with an absolute error near
`eps * |centre| * sqrt(N)` for `eps = 2^-24`. Dividing by N gives the error in
the mean, and normalising divides by the spread, so the output error floor is

```
eps * |centre| * sqrt(N) / spread
```

At centre 4000, spread 1, N 768 that is about 6.6e-3. No fp32 implementation
beats it, so demanding the standard 5e-7 elementwise budget there fails a correct
kernel. The test allows an order of magnitude above the floor, which still sits
two to three orders below what the unstable formula produces.

An earlier version of the test used the flat 5e-7 budget and rejected a known
correct implementation. Testing only the happy path would have shipped it.

### Block size

256 threads divides 768 evenly at 3 elements per thread with no tail. 1024
leaves 256 threads idle on a 768-wide row. 128 halves the warps available to hide
memory latency. Rows narrower than 256 round up to a warp multiple, since a
partial warp wastes lanes inside every shuffle.

### The decode problem

Prefill gives hundreds of rows, hundreds of blocks, and a full GPU. Decode gives
one row, so one block of 256 threads runs on a device with dozens of SMs and the
GPU sits over 99% idle. The kernel becomes launch overhead plus memory latency.

No block size fixes this. The fixes are fusing LayerNorm into the GEMM that
follows it so one launch does both, or capturing the whole decode step in a CUDA
graph to remove per-launch cost. Both come after the naive engine works
end to end.

### Optimisation ladder

| Version | Global reads per row | Status |
| --- | --- | --- |
| Naive, three passes | 3 | done |
| Register-cached, two passes over registers | 1 | not started |
| float4 vectorised load and store | 1 | not started |
| Fused into the following GEMM | 0 | not started |

`rsqrtf` replaces `1.0f / sqrtf` for one instruction instead of several, at about
2 ulp, roughly 2.4e-7 relative. That consumes half the 5e-7 elementwise budget on
its own, so make the swap only after parity is green and re-run the tests.

Numbers go here once each version runs on hardware.
