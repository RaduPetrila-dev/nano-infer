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
| LayerNorm, naive | in progress |
| GELU, embedding, residual | not started |
| GEMM, naive | not started |
| Attention, unfused | not started |
| End-to-end logit parity with HuggingFace | not started |
| BPE tokeniser | not started |
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
skip rather than a failure, so a fresh clone is never red.

## Layout

```
include/nanoinfer/       public interface of the static library
  config.hpp             model dimensions, checkpoint format
  cuda_utils.hpp         RAII for device memory, pinned memory, streams, events
  weights.hpp            loader interface
  device_ops.cuh         warp and block reductions, vectorised access
  kernels/               launcher declarations
src/                     implementation, kernels/ holds the .cu files
tests/                   npy.hpp, reference.hpp, one test per kernel
tools/                   export_gpt2.py, dump_reference.py
bench/                   microbenchmarks and tokens per second
docs/                    kernels.md holds the design and numerical notes
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

## Layout convention

HuggingFace GPT-2 uses `Conv1D`, which stores weights as
`[in_features, out_features]` and computes `y = x @ W`. That matches a row-major
GEMM computing `C[M, N] = A[M, K] * B[K, N]`, so the exporter writes weights
untransposed and the kernels consume them directly. Nothing is transposed
anywhere in this repo.

Kernel sources carry why-only comments. The maths, the numerical traps, the
block configuration and the optimisation ladder live in `docs/kernels.md`.

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
is wrong.

Comparison uses `|a - e| <= absolute + relative * |e|`, the same mixed criterion
as `numpy.allclose`. Pure relative error explodes near zero and post-LayerNorm
activations sit near zero constantly. Pure absolute error is scale-dependent,
and logits reach magnitude 100 while normalised activations sit near 1.

Failures report the worst element ranked by how far past its budget it sits,
with the row and column, not just the flat index. Kernel bugs cluster on an
axis: a whole bad row means the block is wrong, a whole bad column means the
lane is, a scattered handful means a race.

## Benchmarks

Empty until the kernels exist. The table will carry naive, optimised, and
llama.cpp on the same card, with Nsight Compute occupancy and memory throughput
per kernel recorded in `docs/profiling.md`.

## Licence

MIT.
