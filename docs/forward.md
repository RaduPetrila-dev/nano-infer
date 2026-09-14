# Forward pass

`src/model.cpp` wires the kernels into GPT-2. Per-kernel reasoning lives in
`kernels.md`; this file covers the order, the buffers and what the end-to-end
test can and cannot prove.

## Block order

Straight from `GPT2Block.forward`. Two residual branches, each one LayerNorm, a
sub-layer and an add back into the stream.

```
x  = wte[ids] + wpe[positions]

for each block:
    x += attn_proj( attention( qkv( ln_1(x) ) ) )
    x += mlp_proj( gelu( mlp_fc( ln_2(x) ) ) )

logits = ln_f(x) @ wteᵀ
```

Three details in there are worth stating because each is silent when wrong.

The normalisation is pre-layer and not post-layer. GPT-2 normalises the input to
each sub-layer and adds the raw stream back, so `x` is never normalised in place.
Post-layer normalisation is a different model that trains and runs fine and
matches nothing.

The residual is the stream as it entered the branch, not as it entered the block.
Two adds per block, not one.

`ln_f` runs once after the last block, and the head reads `wte` back as its
weight because `lm_head` ties to it. The exporter checks the tie and refuses a
checkpoint without it, so the loader never has to.

## Buffers

One `DeviceArena` sized at construction, so a forward allocates nothing. Each
buffer has exactly one role:

| Buffer | Shape | Holds |
| --- | --- | --- |
| `x` | `[T, d]` | the residual stream, live for the whole pass |
| `norm` | `[T, d]` | LayerNorm output feeding the next projection |
| `qkv` | `[T, 3d]` | packed `[Q \| K \| V]` |
| `attn` | `[T, d]` | merged head output |
| `branch` | `[T, d]` | whichever projection is about to be added into `x` |
| `mlp` | `[T, 4d]` | MLP intermediate, GELU runs over it in place |
| `scores` | `[n_head, T, T]` | attention scratch, probabilities on the way out |
| `logits` | `[T, n_vocab]` | the result |

Reusing `norm` for both LayerNorms and `branch` for both projections is safe
because each is dead by the time it is rewritten, and every launch sits on one
stream, so there is no concurrency to reason about. The aliasing the kernels
allow is used in exactly two places: the residual add writes over the stream it
reads, and GELU runs in place. Everything else is distinct, which is what the
GEMM and attention contracts require.

At 1024 tokens the workspace is 277 MiB, of which the logits are 196 MiB and the
score matrix 48 MiB. The other seven buffers together are 33 MiB. Sizing comes
from `max_tokens` rather than `n_ctx`, so a caller that only ever runs short
prompts does not pay for a full context.

## Why this file is a .cpp

There is no device code in the forward pass, only launches. Compiling it with the
host compiler keeps it out of `nvcc` and gives it the same warning set as the
rest of the library.

## The stage hook

`forward` takes an optional callback and fires it after the embedding, after each
block, after `ln_f` and after the head. Each call synchronises the stream, so it
is a debugging path and never a decode path.

It exists because of what an end-to-end failure looks like without it. Every
kernel passes its own test, the logits are wrong, and nothing says which of
thirteen stages drifted. With it, `tests/test_model.cpp` compares the residual
stream at each boundary the dump records and the first red line names the block.

## Tolerances

The stage checks use `Tolerance::gemm()` and the logits use `Tolerance::logits()`,
both with the absolute term replaced by `relative * rms(reference)`.

The flat term fails a correct engine for the same reason it failed the GEMM.
Error in an activation tracks the sums that produced it, not its own magnitude,
so an element that cancels toward zero inside a tensor whose scale is 1 is
entitled to the error of a full `k = 768` reduction. Scaling the absolute term to
the reference RMS states that directly and needs no chain-depth guesswork.

The embedding is the exception and keeps the elementwise budget. It is one fp32
add of two table rows in the same order PyTorch uses, so it lands on the
reference exactly.

## What the test proves

`test_model.cpp` runs the real prompt through the real checkpoint and checks the
stream at every dumped boundary, the logits, and the argmax at every position.
The argmax check earns its place: logits inside tolerance with a different top
token is still a broken engine, and it is the only assertion in the repo about
what the model actually predicts.

It needs both the exported checkpoint and the reference dump, and skips when
either is missing, so a fresh clone stays green.

## What is not here yet

No KV cache, so each call starts at position zero and recomputes the prompt.
`pos_offset` already runs through the embedding and the attention mask, so the
cache is a buffer and a bookkeeping change rather than a kernel change.

No sampling and no tokeniser, so the CLI takes token ids and prints token ids.

The head computes every row. Generation needs one. At a prompt of 11 tokens the
head is 31% of the forward pass in flop terms, and ten elevenths of that is
discarded, which makes a last-row head the cheapest win still on the table.
