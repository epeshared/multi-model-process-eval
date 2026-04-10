#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate random bf16 safetensors shards for a Hugging Face model dir")
    ap.add_argument("--model-dir", required=True, help="Model directory containing config.json and model.safetensors.index.json")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--force", action="store_true", help="Overwrite existing shard files")
    return ap.parse_args()


def _is_norm_weight(name: str) -> bool:
    return (
        name.endswith("input_layernorm.weight")
        or name.endswith("post_attention_layernorm.weight")
        or name.endswith("norm.weight")
        or name.endswith("q_norm.weight")
        or name.endswith("k_norm.weight")
    )


def _iter_shards(weight_map: Dict[str, str]) -> Iterable[tuple[str, list[str]]]:
    grouped: Dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in weight_map.items():
        grouped[shard_name].append(tensor_name)
    for shard_name in sorted(grouped):
        yield shard_name, sorted(grouped[shard_name])


def _make_tensor(name: str, shape: torch.Size, *, std: float, generator: torch.Generator) -> torch.Tensor:
    if _is_norm_weight(name):
        return torch.ones(shape, dtype=torch.bfloat16)
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    tensor.normal_(mean=0.0, std=std, generator=generator)
    return tensor


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise SystemExit(f"Missing config.json: {config_path}")
    if not index_path.exists():
        raise SystemExit(f"Missing model.safetensors.index.json: {index_path}")

    index_obj = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_obj.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"Invalid or empty weight_map in {index_path}")

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    std = float(getattr(config, "initializer_range", 0.02) or 0.02)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    state = model.state_dict()

    missing = sorted(set(weight_map) - set(state))
    if missing:
        raise SystemExit(f"Index contains tensor names not found in model state_dict, first few: {missing[:10]}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    total_bytes = 0
    for shard_name, tensor_names in _iter_shards(weight_map):
        shard_path = model_dir / shard_name
        if shard_path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing shard without --force: {shard_path}")

        shard_tensors: Dict[str, torch.Tensor] = {}
        shard_bytes = 0
        for tensor_name in tensor_names:
            meta_tensor = state[tensor_name]
            tensor = _make_tensor(tensor_name, meta_tensor.shape, std=std, generator=generator)
            shard_tensors[tensor_name] = tensor
            shard_bytes += tensor.numel() * tensor.element_size()

        save_file(shard_tensors, str(shard_path), metadata={"format": "pt"})
        total_bytes += shard_bytes
        gib = shard_bytes / float(1024**3)
        print(f"wrote {shard_path} tensors={len(shard_tensors)} approx_gib={gib:.3f}")
        del shard_tensors

    print(f"done model_dir={model_dir} total_tensor_bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())