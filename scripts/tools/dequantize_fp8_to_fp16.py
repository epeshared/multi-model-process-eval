#!/usr/bin/env python3
"""Dequantize FP8 (float8) safetensors checkpoints to FP16.

Goal: make a FP8-quantized checkpoint usable in CPU-only environments by
materializing float16 weights.

IMPORTANT
- This is inherently *lossy* because FP8 weights are already quantized.
- This script assumes the common pattern:
    <prefix>.weight:        float8
    <prefix>.weight_scale:  float32 scalar or 1D per-row scale
    <prefix>.input_scale:   float32 (activation scale) [dropped]

It reconstructs an approximate FP16 weight via:
  W_fp16 = W_fp8.to(fp16) * weight_scale

It then drops quantization helper tensors (weight_scale, input_scale) for
weights that were converted.

Example:
  python dequantize_fp8_to_fp16.py \
    --in-model-dir /path/to/fp8_model \
    --out-model-dir /path/to/fp16_model \
    --dtype float16

Notes:
- Requires: torch, safetensors
- Needs extra RAM; output weights are larger than input.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-model-dir", required=True)
    p.add_argument("--out-model-dir", required=True)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-quant-aux", action="store_true", help="Keep *.weight_scale/*.input_scale tensors")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _copy_non_safetensors(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.name.endswith(".safetensors"):
            continue
        out = dst / item.name
        if item.is_dir():
            shutil.copytree(item, out, dirs_exist_ok=True)
        else:
            shutil.copy2(item, out)


def _broadcast_scale(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Common cases:
    # - scalar: ()
    # - per-output-channel: (out_features,)
    # - already broadcastable
    if scale.numel() == 1:
        return scale

    # Try to align 1D scale to the first dim of 2D weight (linear weight).
    if scale.ndim == 1 and weight.ndim == 2 and scale.shape[0] == weight.shape[0]:
        return scale.view(-1, 1)

    # Fall back to raw scale; torch broadcasting will error if incompatible.
    return scale


def main() -> None:
    args = parse_args()

    src = Path(args.in_model_dir).expanduser().resolve()
    dst = Path(args.out_model_dir).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Input model dir does not exist: {src}")

    if dst.exists() and not args.overwrite:
        raise SystemExit(f"Output dir exists: {dst} (use --overwrite)")
    dst.mkdir(parents=True, exist_ok=True)

    out_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Input : {src}")
    print(f"Output: {dst}")
    print(f"Target dtype: {out_dtype}")

    _copy_non_safetensors(src, dst)

    from safetensors.torch import load_file, save_file

    st_files = sorted(src.glob("*.safetensors"))
    if not st_files:
        raise SystemExit("No .safetensors files found")

    total_converted = 0
    total_seen_fp8 = 0

    for st_path in st_files:
        tensors = load_file(str(st_path), device="cpu")

        # Identify fp8 weights
        fp8_weight_keys = [
            k
            for k, t in tensors.items()
            if k.endswith(".weight") and (t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2))
        ]
        total_seen_fp8 += len(fp8_weight_keys)

        converted_prefixes: set[str] = set()
        out_tensors: dict[str, torch.Tensor] = {}

        # Convert fp8 weights
        for k in fp8_weight_keys:
            w_fp8 = tensors[k]
            prefix = k[: -len(".weight")]
            scale_k = prefix + ".weight_scale"

            if scale_k not in tensors:
                raise RuntimeError(f"Missing scale tensor for {k}: expected {scale_k}")

            scale = tensors[scale_k]
            scale_bc = _broadcast_scale(scale, w_fp8)

            w = (w_fp8.to(torch.float16) * scale_bc.to(torch.float16)).to(out_dtype)
            out_tensors[k] = w
            converted_prefixes.add(prefix)
            total_converted += 1

            if args.verbose:
                print(f"[fp8->fp16] {k} {w_fp8.dtype} -> {out_dtype}  scale={scale_k} shape={tuple(w.shape)}")

        # Copy remaining tensors (and optionally drop quant aux)
        for k, t in tensors.items():
            if k in out_tensors:
                continue

            if not args.keep_quant_aux:
                # Drop aux tensors for converted weights
                if k.endswith(".weight_scale"):
                    prefix = k[: -len(".weight_scale")]
                    if prefix in converted_prefixes:
                        continue
                if k.endswith(".input_scale"):
                    prefix = k[: -len(".input_scale")]
                    if prefix in converted_prefixes:
                        continue

            # Cast float8 leftovers defensively (should be rare)
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if args.verbose:
                    print(f"[warn] unexpected float8 tensor not ending with .weight: {k} ({t.dtype}); casting to {out_dtype}")
                out_tensors[k] = t.to(out_dtype)
            else:
                out_tensors[k] = t

        out_path = dst / st_path.name
        print(f"[write] {st_path.name} -> {out_path.name} (converted {len(fp8_weight_keys)} fp8 weights)")
        save_file(out_tensors, str(out_path))

    print(f"Done. FP8 weights seen: {total_seen_fp8}, converted: {total_converted}")


if __name__ == "__main__":
    main()
