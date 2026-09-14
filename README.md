# nano-infer

GPT-2 inference written from scratch in C++ and CUDA. No cuBLAS, no cuDNN, no
PyTorch at runtime. Every kernel here is hand-written, validated against
HuggingFace activations, and profiled with Nsight Compute.

The engine is built naive first, then optimised, and both numbers are kept. A
kernel that is fast but unverified is not finished, and a speedup with no
baseline is not a measurement.

## Status

| Component | State |
| --- | --- |
| Checkpoint format and loader | done |
| Reference harness against HuggingFace | done |
| Device reduction and vector primitives | done |
| LayerNorm, naive | done |
| GELU, embedding, residual | done |
| GEMM, naive | done |
| Attention, unfused | done |
| End-to-end logit parity with HuggingFace | done |
| BPE tokeniser | next |
| KV cache and sampling | not started |
| GEMM, tiled and register-blocked | not started |
| Attention, fused with online softmax | not started |
| Benchmarks against llama.cpp | not started |

## Build

Requires CMake 3.24+, a C++20 compiler, and CUDA 12.x.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake targets the GPU in the build machine by default. `native` queries the
device, so on a machine without one, a compile-only check needs an explicit
architecture:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86;89;90"   # portable binary
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75           # compile check, no GPU present
```

A Codespace can generate reference data, export weights, and compile. It cannot
run the tests, since Codespaces have no GPU.

Options:

| Flag | Default | Effect |
| --- | --- | --- |
| `NANOINFER_LINEINFO` | ON | `-lineinfo`, source correlation in Nsight Compute |
| `NANOINFER_FAST_MATH` | OFF | `-use_fast_math`, changes numerics, breaks parity tests |

`--ptxas-options=-v` is always on, so register counts and spills appear at build
time instead of during a profiler session.

## Weights

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python tools/export_gpt2.py --model gpt2 --out weights/gpt2-124m-f32.bin
./build/nano-infer weights/gpt2-124m-f32.bin
```

Pass token ids after the checkpoint to run one prefill pass and print the top
five next tokens. Ids and not text, since the BPE tokeniser is a later
milestone.

```bash
./build/nano-infer weights/gpt2-124m-f32.bin 464 3139 286 4881 318
```

The CPU-only torch wheel is deliberate. Weight export and the reference dump both
run on CPU, and the default wheel bundles a 2.5 GB NVIDIA runtime neither needs.

## Reference data

Every kernel is checked against the corresponding PyTorch sub-module, not
eyeballed. Generate the activations before running the test suite:

```bash
python tools/dump_reference.py --model gpt2 --out tests/data
```

The dump is deterministic. Fixed prompt, `eval()` mode, `no_grad`, float32 on
CPU. Two machines produce identical files.

Layers 0 and 11 are dumped by default, roughly 7 MiB. All twelve costs about
80 MiB. Layer 0 catches a broken kernel and the last layer catches error that
accumulates up the stack, which covers almost every case. Use `--layers all`
when a bug appears halfway.

The data is gitignored. Tests that need it exit 77, which `ctest` records as a
skip rather than a failure, so a fresh clone is never red. `test_model` needs the
exported checkpoint as well and skips without it, or reads `NANOINFER_CHECKPOINT`
when the file sits somewhere else.

Some kernels get parity for free from tensors the dump already holds. GELU is
checked by running `h.0.mlp.fc.out` through the kernel and comparing against
`h.0.mlp.proj.in`, which is post-activation by construction. Embedding has no
parity case yet, because the dump saves LayerNorm gamma and beta but not `wte`
or `wpe`. It arrives with the end-to-end test, which loads the checkpoint.

Attention gets parity for free as well, and needs no flags. `h.0.attn.qkv.out`
is the packed `[Q | K | V]` the kernel consumes, `h.0.attn.probs` is the
probability matrix it produces, and `h.0.attn.proj.in` is the merged head output
the projection reads, which is the attention result by construction.

GEMM needs the `Conv1D` weights, which the dump writes only when asked:

```bash
python tools/dump_reference.py --dump-weights
```

That costs about 28 MiB per dumped layer and switches on four parity cases, one
per projection in layer 0. Without it they skip. Those cases are the only check
on the layout convention below, since a synthetic case reads back whatever
layout it wrote.

## Layout

```
include/nanoinfer/       public interface of the static library
  config.hpp             model dimensions, checkpoint format
  cuda_utils.hpp         RAII for device memory, pinned memory, streams, events
  weights.hpp            loader interface
  model.hpp              the forward pass
  device_ops.cuh         warp and block reductions, vectorised access
  launch.cuh             block and grid sizing shared by the launchers
  kernels/               launcher declarations
    layernorm.cuh
    gelu.cuh
    embedding.cuh
    residual.cuh
    gemm.cuh
    attention.cuh
src/                     implementation, kernels/ holds the .cu files
tests/                   npy.hpp, reference.hpp, one test per kernel
tools/                   export_gpt2.py, dump_reference.py
bench/                   microbenchmarks and tokens per second
docs/                    kernels.md per kernel, forward.md for the wiring
```

Headers mirror sources one to one. `.cuh` means the file contains device code or
CUDA types, `.hpp` means pure host code.

Shared device helpers live in `device_ops.cuh` as header-only
`__device__ __forceinline__`. A `__device__` function defined in one translation
unit and called from another needs separable compilation, and device linking
blocks cross-file inlining. Warp reductions and vectorised loads sit in the
innermost loop of every kernel, so a real call there would cost more than the
work being done.

## Checkpoint format

One flat file, mapped at load time and uploaded into a single device
allocation.

```
[0, 64)                     header: magic, version, dtype, model dimensions
[64, 64 + 96 * n_tensors)   directory: name, offset, byte count, shape
align to 256                data: every tensor 256-byte aligned
```

Offsets are absolute and pre-aligned, so the relative layout on disk survives
unchanged on the device. One `cudaMalloc` holds every weight.

Upload runs through two pinned staging buffers with event-gated double
buffering. While one chunk crosses PCIe the next is copied out of the page cache
into the other buffer. Copying straight from pageable mapped memory blocks the
calling thread and roughly halves achievable bandwidth.

The directory is fully validated before any GPU allocation happens, so a
malformed file costs nothing.

The format is defined in `include/nanoinfer/config.hpp` and
`tools/export_gpt2.py`. Both sides must change together.

## Forward pass

`src/model.cpp` wires the kernels into GPT-2. Pre-layer normalisation, two
residual branches per block, one `ln_f`, and a head that reads `wte` back as its
weight because `lm_head` ties to it.

Activations live in one arena sized at construction, so a forward allocates
nothing. `tests/test_model.cpp` runs the real prompt through the real checkpoint
and compares the residual stream at every boundary the dump records before it
looks at the logits, so a failure names the block rather than the stack. It also
checks the argmax at every position, which is the only assertion here about what
the model predicts rather than what it computes.

Order, buffer plan and the tolerance rule live in `docs/forward.md`.

## Layout convention

HuggingFace GPT-2 uses `Conv1D`, which stores weights as
`[in_features, out_features]` and computes `y = x @ W`. That matches a row-major
GEMM computing `C[M, N] = A[M, K] * B[K, N]`, so the exporter writes weights
untransposed and the kernels consume them directly. Nothing is transposed
anywhere in this repo.

Kernel sources carry why-only comments. The maths, the numerical traps, the
block configuration and the optimisation ladder live in `docs/kernels.md`.

## Activation

`GPT2Config.activation_function` is `gelu_new`, the tanh approximation, not the
erf definition. The two forms differ by up to 4.7e-4 absolute at `|x| = 2.7`,
which is 99 times the tolerance the GELU test allows and about 900 times the raw
elementwise budget. Substituting one for the other produces a kernel that is
correct in the abstract and wrong here.

## Aliasing

Elementwise kernels run in place. LayerNorm normalises the residual stream over
itself, GELU overwrites the MLP intermediate, and the residual add writes back
into the stream it read. None of those kernels marks its pointers `__restrict__`,
because a restrict-qualified pointer promises the compiler no other pointer
writes the same object, and an in-place launch breaks the promise. All three are
bandwidth bound, so the qualifier buys close to nothing.

## Numerical tolerances

Tolerances come from fp32 error analysis, not from whatever made the test pass.
The mantissa is 24 bits, and summing K products accumulates rounding roughly as
`sqrt(K) * 2^-24`.

| Kernel class | Relative | Reasoning |
| --- | --- | --- |
| Elementwise | 5e-7 | a few operations per element, near one ulp |
| GEMM | 2e-5 | K = 768 gives about 2e-6, an order of magnitude of headroom |
| Attention | 5e-5 | softmax plus a second GEMM |
| Logits | 2e-4 | the whole stack, twelve blocks deep |

These are ceilings for a correct kernel. A kernel that needs a looser tolerance
than the derived floor for its inputs is wrong.

Every row is a ceiling and not a floor. Where the fp32 error floor sits above the
ceiling, the test derives the tolerance from the input instead of relaxing the
number until the test goes green. Three kernels reach that point. LayerNorm at
large mean has a floor of `eps * |centre| * sqrt(N) / spread`, about 6.6e-3 at
centre 4000. GELU in the negative tail has a floor of `|x| * 2^-24`, which is
2.4e-7 at x = -4 against a value of 7.0e-5. GEMM has a floor of
`eps * sqrt(sum of squared partial sums)`, since a dot product accumulates error
in proportion to its partial sums and not to its result. The flat 1e-6 absolute
term misses that by up to 18x on outputs that cancel toward zero, putting 110 of
6144 elements outside tolerance on the `mlp.proj` shape. All three derivations
live in the test that uses them, with the reasoning in `docs/kernels.md`.

Comparison uses `|a - e| <= absolute + relative * |e|`, the same mixed criterion
as `numpy.allclose`. Pure relative error explodes near zero and post-LayerNorm
activations sit near zero constantly. Pure absolute error is scale-dependent,
and logits reach magnitude 100 while normalised activations sit near 1.

The end-to-end test scales the relative term with depth, because a tensor twelve
blocks down the residual stream carries the rounding of every GEMM and softmax
above it and each block hands its error to the next. Block 0 keeps the
single-kernel GEMM figure, and the last block lands next to the logits figure
instead of facing a cliff one LayerNorm before it. Holding the last block to the
single-kernel number while accepting 2e-4 for the logits it feeds was the
inconsistency this replaces.

Failures report the worst element ranked by how far past its budget it sits,
with the row and column, not just the flat index. Kernel bugs cluster on an
axis: a whole bad row means the block is wrong, a whole bad column means the
lane is, a scattered handful means a race.

## Baseline

Measured before any optimisation. Both numbers get kept, so these stay in place
as the optimised kernels land.

512-token prefill, GPT-2 124M, fp32, one NVIDIA A100-SXM4-80GB (sm_80, 108 SMs)
on the University of Manchester CSF3. CUDA 12.6.2, GCC 11.5.0, CMake 3.26.5,
`-DCMAKE_CUDA_ARCHITECTURES=80`, commit `69ed711`. All eight tests pass on this
hardware. Kernel times from `nsys`, median of five runs for the two GEMMs, single
run for the rest.

| Kernel | Time (ms) | Share | Launches |
| --- | --- | --- | --- |
| `gemm_bt` (lm_head) | 45.545 | 57.9% | 1 |
| `gemm` (block projections) | 27.412 | 34.8% | 48 |
| `scores` | 3.903 | 5.0% | 12 |
| `context` | 1.020 | 1.3% | 12 |
| `softmax` | 0.406 | 0.5% | 12 |
| `layernorm` | 0.153 | 0.2% | 25 |
| `gelu` | 0.150 | 0.2% | 12 |
| `residual` | 0.126 | 0.2% | 24 |
| `embedding` | 0.008 | <0.1% | 1 |
| Total | 78.72 | | 147 |

Run-to-run spread across the five runs was 0.03% on `gemm_bt` and 0.22% on
`gemm`. GPU clocks are not locked, since `nvidia-smi -lgc` needs root and this is
a shared facility, so a few percent of drift between sessions is expected.

Nsight Compute on the `mlp.proj` shape, m = 512, n = 768, k = 3072, grid
(24, 64, 1) and block (32, 8, 1), duration 835 us:

| Metric | Value |
| --- | --- |
| L1/TEX throughput | 70.1% |
| SM throughput | 65.2% |
| DRAM throughput | 1.0% |
| L1 hit rate | 90.3% |
| L2 hit rate | 96.6% |
| Achieved occupancy | 84.0% |

The naive GEMM is cache bound on load issue, not DRAM bound. Every value it
loads feeds exactly one output, so the limit is how fast loads can be issued
rather than how fast memory can supply them. Arithmetic intensity is the lever,
which is what shared-memory tiling and register blocking buy. Occupancy at 84%
is not the constraint.

Correctness at 5 tokens, ids `464 3139 286 4881 318`, is a separate run. The
512-token figures above are timing only: the ids are `i % 50257` and the output
carries no meaning.

Profiling detail per kernel goes in `docs/profiling.md` as the optimisation
ladder is climbed. Comparison against llama.cpp waits until it has been run on
this same card.

## Known inefficiencies

Named here rather than buried, since each one is measured and none is fixed yet.

Prefill computes logits at every position and reads one. The head is 57.9% of
runtime at 512 tokens. Restricting it to the final row removes almost all of
that, and the all-rows path stays behind a flag because `test_model` compares
every position.

`gemm_bt` gives one warp per output element, so each output reads a full
k-length row of B and nothing is reused across rows. The kernel scaled 105x for
102x the tokens, which is the signature of zero reuse. Row tiling in shared
memory cuts the redundant traffic by the tile factor.

`gemm` is the same mapping problem one level down: one thread per output, each
walking k alone. Parallelism is tied to n while k stays serial, so narrow shapes
collapse. At m = 512 the n = 768 projections run 4x slower than the n = 3072 one
doing identical work.

`scores_kernel` is quadratic in sequence length: 3.2 us at 5 tokens, 325 us at
512. Invisible at this context, dominant at 2048. This is the case for fusing
attention with an online softmax.

## Licence

MIT.
