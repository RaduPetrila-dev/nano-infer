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

### In-place and __restrict__

`out` aliases `in` whenever a LayerNorm normalises the residual stream in place,
so neither pointer carries `__restrict__`. A restrict-qualified pointer promises
the compiler that no other pointer writes the same object, and an in-place launch
breaks that promise. The access pattern happens to be safe, since each thread
reads and writes the same index, but safe by accident is not a guarantee the
language gives you. The same reasoning appears under GELU.

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

## GELU

```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

### The tanh form is not an optimisation

`GPT2Config.activation_function` defaults to `gelu_new`, which is the tanh
approximation above, not the erf definition `0.5 * x * (1 + erf(x / sqrt(2)))`.
The two are different functions, and picking the wrong one produces a kernel
that is correct in the abstract and off by three orders of magnitude here.

| \|x\| | gap between the two forms |
| --- | --- |
| 2.70 | 4.7e-4 absolute, 1.8e-4 relative |
| 0.00 | 0 |
| 8.00 | below 1e-9 |

The peak sits at |x| = 2.70, in the middle of the range the MLP intermediate
occupies. An erf kernel misses the tolerance in `tests/test_gelu.cu` by 99x and
the raw 5e-7 elementwise budget by about 900x, so at least it fails loudly.

### The negative tail

`1 + tanh(u)` collapses toward zero as x goes negative while `|x|` stays large,
so the product is a small number built from a large one.

| x | 1 + tanh(u) | gelu(x) |
| --- | --- | --- |
| -3 | 7.3e-4 | -3.64e-3 |
| -4 | 3.5e-5 | -7.02e-5 |
| -5 | 1.8e-8 | -2.29e-7 |
| below -5.16 | exactly 0 in fp32 | signed zero |

`tanhf` carries about 2 ulp, so the absolute error near the tail is roughly
`0.5 * |x| * 2 * 2^-24`, which reduces to `|x| * 2^-24`. At x = -4 that is 2.4e-7
absolute against a value of 7.0e-5, a relative error of 3.4e-3. Below x = -5 the
relative error of a correct kernel is unbounded, because the true value goes to
zero faster than the error does.

The consequence for the tests is that only the absolute term of the mixed
criterion does real work in the tail, and the default 1e-7 absolute sits below
the floor. `tests/test_gelu.cu` derives the absolute tolerance from the largest
input magnitude in the case, the same way the LayerNorm test derives its floor.

Measured against a double oracle, a correct fp32 kernel lands at:

| Case | max absolute error | fraction of the derived budget |
| --- | --- | --- |
| normal(0, 2), 8 x 3072 | 4.4e-7 | 0.07 |
| sweep over [-8, 8] | 4.3e-7 | 0.06 |
| tail over [-12, -3] | 1.5e-7 | 0.02 |
| saturation over [-40, 40] | 3.5e-7 | 0.01 |

Every one of those exceeds the 1e-7 absolute default, so the flat elementwise
tolerance rejects a correct GELU before it rejects a wrong one.

Rewriting as `x * sigmoid(2u)` removes the cancellation and is more accurate than
the reference it is checked against, which makes parity worse rather than better.
Match the reference formula.

### Saturation

At large positive x, `tanhf` returns exactly 1 and the result is x. At x below
-5.16 the fp32 sum `1 + tanhf(u)` is exactly 0 and the result is a signed zero.
Both are the correct limits and both match torch. `x + 0.044715 * x^3` at
|x| = 1000 reaches 4.5e10, nowhere near an fp32 overflow, so the residual stream
outlier dimensions pass through without a special case.

### Block configuration

One thread per element, 256 threads per block, grid-stride so the launcher never
inspects grid limits. The kernel moves 8 bytes per element and spends roughly a
dozen instructions on `tanhf`, which compiles to a range reduction and an `ex2`
rather than a library call. Whether it lands memory bound or issue bound is a
question for Nsight, not for a comment.

### In-place and __restrict__

`out` aliases `in` on the real call path, so neither pointer carries
`__restrict__`. A restrict-qualified pointer promises the compiler no other
pointer writes the same object, and an in-place launch breaks that promise. The
kernel is bandwidth bound, so the qualifier buys little here anyway.

The version that wants both is templated on an `InPlace` flag, with the launcher
picking the instantiation from a pointer comparison. Worth doing when the
profile says the aliasing assumption costs something, not before.

### Optimisation ladder

| Version | Bytes per element | Status |
| --- | --- | --- |
| Naive scalar | 8 | done |
| float4 load and store | 8 | not started |
| Fused into the mlp.fc GEMM epilogue | 4 | not started |

Fusing removes the write and the reread entirely, which halves traffic. It is the
only change here worth a real number.

## Embedding

```
out[t][c] = wte[ids[t]][c] + wpe[pos_offset + t][c]
```

### One block per token

Both tables are row-major with d_model contiguous, so a block that owns a token
walks one contiguous run in each table and every load coalesces. Indexing by flat
element instead spreads a single row across several blocks, moves the same bytes,
and loses the locality for nothing.

The gather is fully random in `wte`: 50257 rows of 3 KiB each, and consecutive
tokens land anywhere. Nothing fixes that, and nothing needs to, since prefill
touches at most a few hundred of those rows once.

### pos_offset exists for decode

Prefill passes 0. Decode passes the number of tokens already in the KV cache, so
`wpe` is read at the absolute position rather than at 0 every step. Getting this
wrong gives a model that reads fluently for one token and then loses all sense of
position, which looks like a sampling bug and is not one.

A position past `n_ctx` reads outside `wpe` and returns plausible garbage rather
than crashing, so the launcher rejects it. Token ids are checked with a device
assert, which release builds drop.

### Ties to the output projection

GPT-2 ties `lm_head` to `wte`, and the exporter refuses to write a checkpoint
where they are untied. The final logits GEMM reads the same tensor this kernel
gathers from, so any layout change here changes that GEMM too.

### Optimisation ladder

| Version | Status |
| --- | --- |
| Naive gather, one block per token | done |
| float4 load and store | not started |
| Fused with the first LayerNorm | not started |

## Residual

```
out[i] = a[i] + b[i]
```

One add per 12 bytes moved, an arithmetic intensity of 0.083 flop per byte. Every
GPU worth targeting sits above 30 flop per byte at the roofline knee, so this is
memory bound by more than two orders of magnitude and a scalar grid-stride loop
already runs at close to peak bandwidth. There is no interesting version of this
kernel.

The cost is the launch and the traffic. A 124M forward pass makes 24 of these calls, and at decode each one moves 9 KiB, which is far below the size where a kernel launch pays for itself.

`out` aliases `a` on every call, since the residual stream is updated in place,
so no pointer carries `__restrict__`.

### Optimisation ladder

| Version | Bytes per element | Status |
| --- | --- | --- |
| Naive scalar | 12 | done |
| float4 | 12 | not started |
| Fused into the producing GEMM epilogue | 8 | not started |
| Absorbed by a CUDA graph capture of the decode step | 8 | not started |

Only the last two matter. The float4 row exists to confirm the kernel was already
bandwidth limited.
