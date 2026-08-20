#!/usr/bin/env python3
"""Dump HuggingFace GPT-2 intermediate activations as .npy reference data.

    pip install torch transformers numpy
    python tools/dump_reference.py --model gpt2 --out tests/data

Every kernel in nano-infer gets validated against the tensor this script saves
for the matching sub-module. Without it you find out your GEMM is wrong twelve
files later, with no way to tell which layer introduced the error.

Determinism matters more than realism here. The prompt is fixed, dropout is off
in eval mode, and the model runs under no_grad on CPU in float32. Run this on
two machines and you get byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Short enough to keep the dump small, long enough that causal masking, position
# embeddings and multi-token attention all get exercised. A single token would
# let a broken mask pass every test.
DEFAULT_PROMPT = "The capital of France is Paris, and the capital of Italy is"


def resolve_layers(spec: str, n_layer: int) -> list[int]:
    """Parse a layer spec like '0,-1' or 'all' into concrete indices.

    Dumping all 12 layers costs about 80 MiB, too much for git. Layer 0 and the
    last layer catch almost everything: layer 0 finds a wrong kernel, and the
    last layer finds error that accumulates across the stack. Pass --layers all
    when you are hunting a bug that appears halfway up.
    """
    if spec.strip() == "all":
        return list(range(n_layer))

    out: list[int] = []
    for part in spec.split(","):
        index = int(part.strip())
        if index < 0:
            index += n_layer
        if not 0 <= index < n_layer:
            raise ValueError(f"layer {part} is outside 0..{n_layer - 1}")
        if index not in out:
            out.append(index)
    return sorted(out)


def hooked_modules(model, layers: list[int]) -> list[tuple[str, object]]:
    """Sub-modules whose input and output map onto a nano-infer kernel."""
    tf = model.transformer
    pairs: list[tuple[str, object]] = []
    for i in layers:
        block = tf.h[i]
        pairs += [
            # Input is the residual stream entering the block, output feeds QKV.
            (f"h.{i}.ln_1", block.ln_1),
            # Conv1D producing packed [Q | K | V].
            (f"h.{i}.attn.qkv", block.attn.c_attn),
            # The input here is the merged head output, which is exactly what
            # your fused attention kernel must produce. Hooking c_proj gives you
            # the attention result and the projection result from one hook.
            (f"h.{i}.attn.proj", block.attn.c_proj),
            # Output is the whole block's attention branch, post-residual-add in
            # the block forward, so compare against your block, not your kernel.
            (f"h.{i}.attn", block.attn),
            (f"h.{i}.ln_2", block.ln_2),
            (f"h.{i}.mlp.fc", block.mlp.c_fc),
            # Input is post-GELU, output is the second projection.
            (f"h.{i}.mlp.proj", block.mlp.c_proj),
            (f"h.{i}.mlp", block.mlp),
            (f"h.{i}", block),
        ]
    pairs.append(("ln_f", tf.ln_f))
    return pairs


def to_array(value):
    """Unwrap whatever a module returned into a contiguous float32 array."""
    import torch

    if isinstance(value, (tuple, list)):
        # GPT2Attention and GPT2Block return tuples. Element 0 is the tensor.
        value = value[0]
    if not isinstance(value, torch.Tensor):
        return None
    return np.ascontiguousarray(value.detach().to(torch.float32).cpu().numpy())


def dump(model_name: str, out_dir: Path, prompt: str, layer_spec: str) -> None:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"loading {model_name}", file=sys.stderr)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float32)
    # eval() disables dropout. Without this the reference changes every run and
    # every kernel test becomes flaky.
    model.eval()

    cfg = model.config
    layers = resolve_layers(layer_spec, cfg.n_layer)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    n_tokens = int(input_ids.shape[1])

    saved: dict[str, list[int]] = {}

    def save(name: str, array: np.ndarray) -> None:
        if array is None:
            return
        path = out_dir / f"{name}.npy"
        np.save(path, array)
        saved[name] = list(array.shape)

    captured: dict[str, tuple] = {}

    def make_hook(name: str):
        def hook(_module, inputs, output):
            captured[name] = (inputs, output)

        return hook

    handles = []
    for name, module in hooked_modules(model, layers):
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            result = model(input_ids, output_attentions=True)
    finally:
        # Hooks stay attached to the module otherwise, and a second call in the
        # same process would double-fire them.
        for handle in handles:
            handle.remove()

    save("tokens", input_ids.to(torch.float32).numpy())

    # The embedding sum is not a module, so reconstruct it. This is the input to
    # your very first LayerNorm and the one tensor a hook cannot reach.
    with torch.no_grad():
        positions = torch.arange(n_tokens, dtype=torch.long).unsqueeze(0)
        embedded = model.transformer.wte(input_ids) + model.transformer.wpe(positions)
    save("embed.out", to_array(embedded))

    for name, (inputs, output) in captured.items():
        if inputs:
            save(f"{name}.in", to_array(inputs[0]))
        save(f"{name}.out", to_array(output))

    # Attention probabilities, post-softmax and post-mask, shaped
    # [batch, heads, query, key]. The single most useful tensor for debugging a
    # fused attention kernel, because it isolates the softmax from the two GEMMs
    # around it.
    if result.attentions is not None:
        for i in layers:
            save(f"h.{i}.attn.probs", to_array(result.attentions[i]))

    save("logits", to_array(result.logits))

    manifest = {
        "model": model_name,
        "prompt": prompt,
        "n_tokens": n_tokens,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "n_ctx": cfg.n_positions,
        "n_vocab": cfg.vocab_size,
        "layers_dumped": layers,
        "torch_version": torch.__version__,
        "tensors": saved,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_mib = sum(
        (out_dir / f"{n}.npy").stat().st_size for n in saved
    ) / (1024 * 1024)
    print(
        f"wrote {len(saved)} tensors to {out_dir} ({total_mib:.1f} MiB), "
        f"{n_tokens} tokens",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model id")
    parser.add_argument("--out", type=Path, default=Path("tests/data"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--layers",
        default="0,-1",
        help="comma-separated layer indices, negatives count from the end, or 'all'",
    )
    args = parser.parse_args()

    dump(args.model, args.out, args.prompt, args.layers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
