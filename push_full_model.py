"""Merge a PEFT/LoRA checkpoint into a full model and push to Hugging Face.

This repo's GRPO training checkpoints are PEFT adapters (e.g. adapter_model.safetensors).
Competitions often require a *single* Hugging Face model repo that already contains the
merged weights (no separate adapter repo).

Example:
  python push_full_model.py \
    --adapter_dir models/chess-grpo-sequences/checkpoint-2000 \
    --repo_id <your-username>/<repo-name> \
    --dtype bf16

Notes:
- You must be logged in: `huggingface-cli login` (or set HF_TOKEN).
- By default, we try to merge into the non-4bit base model (recommended), inferred from
  adapter_config.json. You can override with --base_model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _read_adapter_base_model(adapter_dir: Path) -> str | None:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None
    data = json.loads(cfg_path.read_text())
    return data.get("base_model_name_or_path")


def _default_base_candidates(base: str) -> list[str]:
    # Prefer merging into a real (bf16/fp16) base model.
    # Many training runs use *-bnb-4bit* variants for efficiency.
    candidates = [base]
    if base.endswith("-bnb-4bit"):
        candidates.insert(0, base[: -len("-bnb-4bit")])
    return list(dict.fromkeys(candidates))


def _torch_dtype_from_arg(dtype: str) -> torch.dtype:
    dtype = dtype.lower().strip()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--max_shard_size", type=str, default="2GB")
    p.add_argument("--private", action="store_true")
    p.add_argument("--no_push", action="store_true", help="Only save locally; do not push.")
    args = p.parse_args()

    if args.repo_id is None:
        args.no_push = True

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise SystemExit(f"adapter_dir not found: {adapter_dir}")

    inferred_base = _read_adapter_base_model(adapter_dir)
    base_model = args.base_model or inferred_base
    if not base_model:
        raise SystemExit(
            "Could not infer base model from adapter_config.json. Provide --base_model."
        )

    dtype = _torch_dtype_from_arg(args.dtype)

    # Use GPU if available; merging a small 0.5B model is fine on a single A6000.
    device_map = {"": 0} if torch.cuda.is_available() else None

    # Load tokenizer from adapter checkpoint if present; otherwise from base.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Try preferred base candidates first.
    last_err: Exception | None = None
    model = None
    chosen_base = None
    for candidate in _default_base_candidates(base_model):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            chosen_base = candidate
            break
        except Exception as e:
            last_err = e

    if model is None or chosen_base is None:
        raise SystemExit(f"Failed to load base model. Last error: {last_err}")

    # Load adapter and merge.
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    merged = model.merge_and_unload()

    # Save locally.
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = adapter_dir.parent / (f"merged-{adapter_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged.save_pretrained(
        str(out_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(out_dir))

    print("Merged model saved:")
    print(f"  base:    {chosen_base}")
    print(f"  adapter: {adapter_dir}")
    print(f"  out:     {out_dir}")

    if args.no_push:
        return

    # Push to HF.
    print(f"Pushing to Hugging Face: {args.repo_id} (private={args.private})")
    merged.push_to_hub(
        args.repo_id,
        private=args.private,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.push_to_hub(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
