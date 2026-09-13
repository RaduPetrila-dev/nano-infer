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

## GEMM

```
C[m, n] = A[m, k] * B[k, n] + bias[n]
```

Four calls per transformer block, and every weight in the model except `wte`
passes through one of them.

| Call | m | k | n |
| --- | --- | --- | --- |
| `attn.qkv` | tokens | 768 | 2304 |
| `attn.proj` | tokens | 768 | 768 |
| `mlp.fc` | tokens | 768 | 3072 |
| `mlp.proj` | tokens | 3072 | 768 |

### Nothing transposes

HuggingFace GPT-2 stores every projection as `Conv1D`, which holds its weight as
`[in_features, out_features]` and computes `y = x @ W`. A row-major
`C[m, n] = A[m, k] * B[k, n]` wants exactly that, so the exporter writes the
weight untransposed and the kernel reads it as `B` directly.

This is the one property a synthetic test cannot check. A generated case writes
whatever layout it then reads back, so a kernel that transposed both the write
and the read would pass every shape in the file. The parity cases exist for this
alone: real layer 0 activations, real layer 0 weights, compared against what the
`Conv1D` produced. Run `tools/dump_reference.py --dump-weights` to enable them.

### Thread mapping

One thread per output element, a 32 by 8 block tile.

32 columns wide is the load that matters. Lane `i` of a warp computes column
`col0 + i` of the same output row, so on every step of the reduction the 32 lanes
read 32 consecutive floats of one `B` row. That is 128 bytes, one transaction.
Any other tile width splits it into two.

The `A` access is the mirror case. Every thread in the warp shares a row, so all
32 lanes read the same address and the load broadcasts rather than costing
bandwidth. Eight rows of tile stack eight warps and reach the 256 threads every
other kernel here uses.

The grid rounds up to whole tiles, so edge blocks carry threads with no output.
They return immediately. No barrier follows the guard, which is what makes the
early return legal here and illegal in LayerNorm.

### Arithmetic intensity, and why naive is still worth shipping

A tile of `Tm` by `Tn` outputs loads `(Tm + Tn) * k` floats and does
`2 * Tm * Tn * k` flops, so ignoring cache the intensity is

```
Tm * Tn / (2 * (Tm + Tn))   flop per byte
```

At 8 by 32 that is 3.2. The roofline knee on current cards sits between about 10
and 80 flop per byte for fp32, so this kernel is memory bound everywhere, by a
factor of three at the low end.

The gap that justifies the tiled milestone shows up at long prefill. At 1024
tokens the `attn.qkv` call is 3.62 GFLOP against 19.7 MiB of distinct operand
data, an achievable 184 flop per byte. The naive kernel launches 9216 blocks that
between them pull 1.13 GiB, which is 57 times the traffic the arithmetic needs.

The gap at short prefill is the surprise, and it is worth knowing before anyone
optimises the wrong thing. At 8 tokens the same call touches 7.08 MiB of weight
for 28 MFLOP, so a perfect GEMM reaches 4 flop per byte and the naive kernel
already reaches 3.2. Tiling buys almost nothing there. At decode, `m` is 1, the
GEMM is a GEMV, and the intensity is 0.5 flop per byte no matter what any kernel
does: the whole weight has to cross the bus to produce one output row. The fixes
for decode are a smaller dtype and batching, not a better tile.

### fmaf, and where the bias goes

The inner loop uses `fmaf` rather than `a * b + acc`, matching `dot4` in
`device_ops.cuh`. Two reasons. The numerics stop depending on whether the
compiler contracts the pair, and one rounding per term instead of two roughly
halves the accumulated error, which is worth having at `k = 3072`.

The bias is added once at the end, outside the reduction. Seeding the accumulator
with it instead would put it at the head of a 3072-long dependency chain and
round it 3072 times rather than once. `bias` may be null, and the branch is
uniform across the entire grid, so it costs nothing.

### The absolute tolerance has to be derived

`Tolerance::gemm()` is `{2e-5 relative, 1e-6 absolute}`. The relative half is
right. The absolute half fails a correct kernel, and this is the third kernel in
this file where that happens.

A dot product of `k` terms accumulates absolute error set by the size of its
partial sums, not by the size of its result. Summing sequentially rounds once per
step against the partial sum `S_j`, and those roundings add as a random walk, so

```
floor = eps * sqrt( sum_j S_j² )       eps = 2^-24
```

When the terms are zero-mean the partial sums wander to about `sqrt(j)` and the
result lands near `sqrt(k)`, while the error keeps growing with `k`. Outputs that
cancel down to near zero are correct to every bit they are entitled to and still
miss a flat 1e-6 by an order of magnitude.

Measured against a sequential `fmaf` reference, the flat budget fails five of the
ten synthetic shapes in `tests/test_gemm.cu`. The `mlp.proj` shape puts 110 of
6144 outputs outside it, the worst by 18x.

The floor above costs one multiply-add per element to track, since the oracle is
already walking `j`. Measured worst-case error lands between 0.4 and 1.2 times
it across every shape tested, so the 10x margin the test applies is a margin.
The worst case in the suite then sits at 7% of its budget, and a transposed
operand or a dropped term misses by the magnitude of the result itself, four
orders above.

### The output head is the transposed case

`lm_head` ties to `wte`, which is `[n_vocab, d_model]`, so the logits contract
against a stored row rather than a stored column. That needs a transposed-`B`
variant and arrives with the end-to-end logit test, where there is something to
check it against.

### Optimisation ladder

| Version | flop per byte of tile load | Status |
| --- | --- | --- |
| Naive, 8 by 32, one thread per output | 3.2 | done |
| Shared-memory tile, 32 by 32 | 8 | not started |
| Register blocked, 64 by 64 at 4 by 4 per thread | 16 | not started |
| 128 by 128 with float4 loads and a double-buffered tile | 32 | not started |
| Fused bias, GELU and residual epilogues | no change | not started |

The column ignores L2, which recovers a real share of the repeated loads, so
measured numbers should beat it. The ordering is what the column is for.

The epilogue row changes no intensity and still matters. Folding the bias, the
GELU and the residual add into the GEMM that produces their input removes three
full read-write cycles over the activation per block, twenty-four times per
forward pass, and those kernels are pure bandwidth.

Numbers go here once each version runs on hardware.

## Attention, unfused

```
S[h][i][j] = (Q[i][h] · K[j][h]) / sqrt(head_dim)   for j <= i, masked otherwise
P[h][i]    = softmax(S[h][i])
O[i][h]    = Σ_j P[h][i][j] · V[j][h]
```

Heads never interact, so `h` is a grid dimension and never a loop.

### Three kernels, one matrix

QKᵀ, the row softmax and the value sum run as three launches with the score
matrix written to global memory between them. Fusing them is the endpoint, not
the start. The unfused version exists because it is the only one that can be
checked a stage at a time: `attention_scores` alone produces a matrix you can
diff against `h.0.attn.probs` before the softmax has touched it, and a fused
kernel that is wrong gives you one number and no way in.

It is also the baseline. A fused attention kernel with no unfused measurement
next to it is a claim, not a result.

### Pointers and strides, not one tensor

The QKV projection writes one row of `[Q | K | V]`, 2304 wide for the 124M
model. A KV cache stores K and V on their own, 768 wide, and Q arrives from a
different allocation entirely. Those are the same computation with different
strides, so `AttentionShape` carries a base pointer and a row stride per operand
instead of assuming the packed layout. `AttentionShape::packed` builds the
prefill case, and decode changes three integers rather than forking the kernels.

### The scale goes before the softmax

`1 / sqrt(head_dim)` multiplies the scores, not the probabilities. Softmax is
invariant under an additive shift and not under a multiplicative one, so moving
the factor past it changes the distribution. At `head_dim` 64 the factor is
0.125, a power of two and exact in fp32.

### The mask, and the half that is never computed

Query row `r` sits at absolute position `pos_offset + r` and sees keys `0`
through `pos_offset + r`. Prefill passes offset 0. Decode passes one query row
with the offset set to the cache length, and the same expression covers both.
Without the offset every decode step attends to key 0 alone, which is the same
class of bug `pos_offset` exists to prevent in the embedding kernel.

The score kernel writes `-inf` into masked positions so the intermediate matrix
is defined everywhere and worth dumping. The softmax overwrites those with exact
zeros, matching what HuggingFace reports, and both later kernels stop their loops
at the row's visible length. Storage is square, work is triangular.

An off-by-one here is the cheapest line in the file to get wrong. Admitting one
future key shifts a prefill row by a fraction of a percent, which reads as
accumulated error rather than as a bug, and it disappears entirely during decode
because there is no future key to leak. `tests/test_attention.cu` checks the mask
structurally, not through a tolerance: masked weights must be exactly zero and
every row must sum to one.

### The max subtraction is not optional

`expf` overflows in fp32 above `ln(FLT_MAX)`, which is 88.72. GPT-2 scores pass
that on real prompts, and an overflowed row computes `inf / inf` and returns NaN
across every key. Subtracting the row max bounds every exponent at zero. The
largest term becomes exactly 1 and the smallest underflows to zero, which is the
right answer for a term the denominator was going to ignore.

The cost is one extra block reduction. The `overflow_2h_6x64` case builds scores
of exactly 96.0 to 103.5 from dyadic constants, so the dot product is
bit-identical in fp32 and in the double oracle and the only thing under test is
the subtraction. Remove it and the case returns NaN for most of the matrix.

### One warp per key

The score kernel gives a block to each `(query, head)` and a warp to each key.
Lanes stride the head dimension, so the 32 addresses a warp issues are
contiguous and the load retires as one transaction. One thread per key instead
puts neighbouring threads a full row apart, 9216 bytes for a packed row, and
turns each load into 32 separate transactions.

The key loop is `for (key = warp; key < n_key; key += warps)`. The bound depends
on the warp index and never on the lane, so all 32 lanes make the same number of
trips. `warp_reduce_sum` shuffles under the full mask and reads undefined data
from any lane that left early, which is the failure mode a lane-dependent bound
would produce: no error, no hang, just a wrong sum on the tail warp.

The query row stages into dynamic shared memory once per block, since all eight
warps read it and it is 256 bytes at `head_dim` 64.

### Reduction identities

The softmax kernel runs one block per row and reuses one shared array for the
max and the sum, which `block_reduce_sum` already supports through its second
barrier.

`block_reduce_max` seeds lanes past the warp count with `-FLT_MAX` rather than
`-inf`. A row with no visible key would take that finite value as its maximum,
exponentiate to zero, and divide by a zero sum. Causal masking puts the diagonal
in every row, so the case cannot arise, and `attention_validate` rejects any
shape where `n_key` fails to reach the last query position rather than leaving
the invariant implicit.

### The context kernel

One block per `(query, head)`, one thread per output channel, each thread
walking the visible keys. Neighbouring threads read neighbouring channels of the
same value row, so the loads coalesce and the probability broadcasts across the
block. At `head_dim` 64 the block is two warps, which is thin, and the grid is
`n_query * n_head` blocks, which is not. The optimised version splits the keys
across warps and reduces, which only pays once the kernel stops being a naive
baseline.

### What the matrix costs

The score buffer is `n_head * n_query * n_key` floats. At full context that is
12 · 1024 · 1024 · 4 bytes, 48 MiB per sequence, written once and read twice.
Weights are 475 MiB in fp32, so one sequence at full context adds a tenth of the
model again in scratch, and every byte of it exists only to be read back twice.
That is the argument for flash attention in one line.

### Parity is the only check on the head layout

Nothing reshapes. A head is the column offset `h * head_dim` into a row, so
splitting the heads and merging them again are both address arithmetic and
neither moves a byte, which keeps the repo's claim that nothing is transposed
anywhere true here too.

A synthetic case reads back whatever layout it wrote, so it cannot catch heads
sliced the wrong way. Contiguous slices of the row against an interleaved
reading are both self-consistent and only one matches `Conv1D`. The layer 0
parity case is what settles it, the same argument that makes the GEMM parity
case the only check on the weight transpose.

### The decode problem

Decode gives `n_query` 1, so the grid is `n_head` blocks and the device sits
idle in exactly the way LayerNorm does at one row. The fix is the same: fuse the
stages so one launch does the work, and capture the step in a CUDA graph to drop
the per-launch cost. Neither belongs before the engine runs end to end.

### Optimisation ladder

| Version | Passes over the score matrix | Status |
| --- | --- | --- |
| Three kernels, matrix in global memory | 3 | done |
| float4 loads in the dot product and the value sum | 3 | not started |
| Softmax fused into the score kernel, matrix tiled in shared memory | 1 | not started |
| Online softmax, matrix never leaves registers | 0 | not started |

The last row is flash attention. It removes the 48 MiB and the two extra passes
together, and it is the reason this version keeps its numbers.

Numbers go here once each version runs on hardware.
