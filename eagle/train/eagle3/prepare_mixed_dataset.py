#!/usr/bin/env python
"""
prepare_mixed_dataset.py
========================
Utility for assembling the EAGLE-3 training corpus by mixing UltraChat-200k
and ShareGPT conversations.

The training pipeline under ``eagle/train/eagle3`` expects a ShareGPT-style
JSON Lines (JSONL) file where each line is a dictionary with keys::

    {
        "id": str,
        "conversations": [
            {"from": "human", "value": str},
            {"from": "gpt",   "value": str},
            ...
        ]
    }

Both source datasets are available on the Hugging Face Hub:

* UltraChat-200k – ``HuggingFaceH4/ultrachat_200k``
* ShareGPT (Vicuna cleaned) – ``anon8231489123/ShareGPT_Vicuna_unfiltered``

This script downloads the two datasets (streaming-friendly) and writes the
combined records to ``eagle/data/eagle3_mixed_ultra_sharegpt.jsonl`` so that
other users can reproduce the training data without relying on file paths
specific to the original author.

Usage
-----
Run once before launching training:

>>> python -m eagle.train.eagle3.prepare_mixed_dataset  # defaults are fine

Optionally specify a different output path:

>>> python -m eagle.train.eagle3.prepare_mixed_dataset --out path/to/file.jsonl

The script is idempotent: if the output file already exists it will refuse to
overwrite it unless ``--overwrite`` is passed.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
# Conversion helpers
# ---------------------------------------------------------------------------- #

def _convert_ultrachat(example: dict) -> dict:
    """Convert a single UltraChat example into ShareGPT format."""
    messages = []
    for msg in example.get("messages", []):
        role = msg.get("role")
        if role == "user":
            from_ = "human"
        elif role == "assistant":
            from_ = "gpt"
        else:
            # Skip system/other roles for consistency with ShareGPT style
            continue
        messages.append({"from": from_, "value": msg.get("content", "")})

    return {"id": str(uuid.uuid4()), "conversations": messages}


def _convert_sharegpt(example: dict) -> dict:
    """ShareGPT dataset is *already* in the desired format; just normalise."""
    return {
        "id": example.get("id", str(uuid.uuid4())),
        "conversations": example["conversations"],
    }


# ---------------------------------------------------------------------------- #
# Main entry
# ---------------------------------------------------------------------------- #

def build_dataset(out_path: Path, overwrite: bool = False) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists – delete it or pass --overwrite to regenerate"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ultra_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train", streaming=False)
    share_ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=False)

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in tqdm(ultra_ds, desc="UltraChat 200k"):
            rec = _convert_ultrachat(ex)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for ex in tqdm(share_ds, desc="ShareGPT"):
            rec = _convert_sharegpt(ex)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Mixed dataset written to {out_path.resolve()}")


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #

def cli() -> None:
    parser = argparse.ArgumentParser(description="Create the mixed EAGLE-3 training dataset")
    parser.add_argument(
        "--out",
        type=str,
        default="eagle/data/eagle3_mixed_ultra_sharegpt.jsonl",
        help="Destination JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    args = parser.parse_args()

    build_dataset(Path(args.out), overwrite=args.overwrite)


if __name__ == "__main__":
    cli()