#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights into the nano-infer checkpoint format.

    pip install torch transformers
    python tools/export_gpt2.py --model gpt2 --out weights/gpt2-124m-f32.bin

Layout note. HuggingFace GPT-2 uses Conv1D, which stores its weight as
[in_features, out_features] and computes y = x @ W. That is already the layout a
row-major GEMM wants for C[M, N] = A[M, K] * B[K, N], so no transpose happens
here. The exporter writes exactly what the kernels read.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x494E414E  # "NANI"
VERSION = 1
HEADER_BYTES = 64
ENTRY_BYTES = 96
NAME_BYTES = 48
ALIGN = 256

DTYPES = {
    "f32": (0, np.float32),
    "f16": (1, np.float16),
    "bf16": (2, None),  # written from torch, numpy has no native bfloat16
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def build_name_map(n_layer: int) -> list[tuple[str, str]]:
    """Ordered list of (huggingface name, nano-infer name)."""
    pairs = [
        ("transformer.wte.weight", "wte"),
        ("transformer.wpe.weight", "wpe"),
    ]
    for i in range(n_layer):
        hf = f"transformer.h.{i}."
        ni = f"h.{i}."
        pairs += [
            (hf + "ln_1.weight", ni + "ln_1.w"),
            (hf + "ln_1.bias", ni + "ln_1.b"),
            (hf + "attn.c_attn.weight", ni + "attn.qkv.w"),
            (hf + "attn.c_attn.bias", ni + "attn.qkv.b"),
            (hf + "attn.c_proj.weight", ni + "attn.proj.w"),
            (hf + "attn.c_proj.bias", ni + "attn.proj.b"),
            (hf + "ln_2.weight", ni + "ln_2.w"),
            (hf + "ln_2.bias", ni + "ln_2.b"),
            (hf + "mlp.c_fc.weight", ni + "mlp.fc.w"),
            (hf + "mlp.c_fc.bias", ni + "mlp.fc.b"),
            (hf + "mlp.c_proj.weight", ni + "mlp.proj.w"),
            (hf + "mlp.c_proj.bias", ni + "mlp.proj.b"),
        ]
    pairs += [
        ("transformer.ln_f.weight", "ln_f.w"),
        ("transformer.ln_f.bias", "ln_f.b"),
    ]
    return pairs


def to_numpy(tensor, dtype_name: str) -> np.ndarray:
    import torch

    if dtype_name == "bf16":
        arr = tensor.detach().to(torch.bfloat16).contiguous()
        # Reinterpret the raw 16-bit payload. The loader reads it as bf16.
        return arr.view(torch.uint16).cpu().numpy()

    np_dtype = DTYPES[dtype_name][1]
    return tensor.detach().float().cpu().numpy().astype(np_dtype, copy=False)


def export(model_name: str, out_path: Path, dtype_name: str) -> None:
    import torch
    from transformers import GPT2LMHeadModel

    print(f"loading {model_name}", file=sys.stderr)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    cfg = model.config
    state = dict(model.state_dict())

    pairs = build_name_map(cfg.n_layer)
    tensors: list[tuple[str, np.ndarray]] = []

    for hf_name, ni_name in pairs:
        if hf_name not in state:
            raise KeyError(f"{hf_name} is absent from the state dict")
        if len(ni_name) >= NAME_BYTES:
            raise ValueError(f"name '{ni_name}' does not fit in {NAME_BYTES} bytes")
        arr = np.ascontiguousarray(to_numpy(state[hf_name], dtype_name))
        if arr.ndim == 0 or arr.ndim > 4:
            raise ValueError(f"{ni_name} has rank {arr.ndim}")
        tensors.append((ni_name, arr))

    # lm_head ties to wte in GPT-2, so it is not written twice.
    tied = torch.equal(state["lm_head.weight"], state["transformer.wte.weight"])
    if not tied:
        raise ValueError("lm_head is untied from wte, the loader assumes tying")

    dtype_tag = DTYPES[dtype_name][0]
    dir_end = HEADER_BYTES + ENTRY_BYTES * len(tensors)
    cursor = align_up(dir_end, ALIGN)

    entries = []
    for name, arr in tensors:
        dims = list(arr.shape) + [0] * (4 - arr.ndim)
        entries.append(
            struct.pack(
                f"<{NAME_BYTES}sQQI4I3I",
                name.encode("ascii"),
                cursor,
                arr.nbytes,
                arr.ndim,
                *dims,
                0,
                0,
                0,
            )
        )
        cursor = align_up(cursor + arr.nbytes, ALIGN)

    header = struct.pack(
        "<9I7I",
        MAGIC,
        VERSION,
        dtype_tag,
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
        cfg.n_positions,
        cfg.vocab_size,
        len(tensors),
        *([0] * 7),
    )
    assert len(header) == HEADER_BYTES, len(header)
    assert all(len(e) == ENTRY_BYTES for e in entries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(header)
        for entry in entries:
            fh.write(entry)

        written = dir_end
        for (name, arr), entry in zip(tensors, entries):
            offset = struct.unpack_from("<Q", entry, NAME_BYTES)[0]
            fh.write(b"\0" * (offset - written))
            fh.write(arr.tobytes())
            written = offset + arr.nbytes

    total_mib = out_path.stat().st_size / (1024 * 1024)
    print(
        f"wrote {out_path} ({len(tensors)} tensors, {dtype_name}, {total_mib:.1f} MiB)",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model id")
    parser.add_argument("--out", type=Path, required=True, help="output .bin path")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="f32")
    args = parser.parse_args()

    export(args.model, args.out, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
