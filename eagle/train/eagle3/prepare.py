#!/usr/bin/env python
"""
prepare.py
========================
Utility for assembling the EAGLE-3 training corpus by mixing UltraChat-200k
and ShareGPT conversations.

The training pipeline under ``eagle/train/eagle3`` expects a ShareGPT-style
JSON Lines (JSONL) file where each line is a dictionary with keys::

    {'id': str, 'conversations': [{'from': 'human', 'value': str}, {'from': 'gpt', 'value': str}, ...]}

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

>>> python -m eagle.train.eagle3.prepare  # defaults are fine

Optionally specify a different output path:

>>> python -m eagle.train.eagle3.prepare --out path/to/file.jsonl

The script is idempotent: if the output file already exists it will refuse to
overwrite it unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
import random

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
# Conversion helpers
# ---------------------------------------------------------------------------- #


def _convert_ultrachat(example: dict) -> dict:
  """Convert a single UltraChat example into ShareGPT format."""
  messages = []
  for msg in example.get('messages', []):
    role = msg.get('role')
    if role == 'user':
      from_ = 'human'
    elif role == 'assistant':
      from_ = 'gpt'
    else:
      # Skip system/other roles for consistency with ShareGPT style
      continue
    messages.append({'from': from_, 'value': msg.get('content', '')})

  return {'id': str(uuid.uuid4()), 'conversations': messages}


def _convert_sharegpt(example: dict) -> dict:
  """ShareGPT dataset is *already* in the desired format; just normalise."""
  conversations = []
  for c in example['conversations']:
    conversations.append({'from': c['from'], 'value': c['value']})
  return {'id': example.get('id', str(uuid.uuid4())), 'conversations': conversations}


# ---------------------------------------------------------------------------- #
# Main entry
# ---------------------------------------------------------------------------- #


def build_dataset(
  train_path: Path, overwrite: bool = False, test_path: Path | None = None, test_ratio: float = 0.0
) -> None:
  """Create mixed dataset and optional test split.

  Parameters
  ----------
  train_path : Path
      Destination file for the training set.
  overwrite : bool, optional
      Overwrite existing files if *True*.
  test_path : Path | None, optional
      If provided, write a held-out test subset to this file.
  test_ratio : float, optional
      Proportion of examples to sample for the test set (0–1).
  """

  if train_path.exists() and not overwrite:
    raise FileExistsError(f'{train_path} already exists – delete it or pass --overwrite to regenerate')
  if test_path is not None and test_path.exists() and not overwrite:
    raise FileExistsError(f'{test_path} already exists – delete it or pass --overwrite to regenerate')

  train_path.parent.mkdir(parents=True, exist_ok=True)
  if test_path is not None:
    test_path.parent.mkdir(parents=True, exist_ok=True)

  # deterministic randomness
  random.seed(42)

  ultra_ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=False)
  ultra_test = load_dataset('HuggingFaceH4/ultrachat_200k', split='test_sft', streaming=False)
  share_ds = load_dataset(
    'anon8231489123/ShareGPT_Vicuna_unfiltered',
    data_files=['ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'],
    split='train',
    streaming=False,
  )

  # open files
  fout_train = train_path.open('w', encoding='utf-8')
  fout_test = None
  if test_path is not None and test_ratio > 0.0:
    fout_test = test_path.open('w', encoding='utf-8')

  def _write(rec: dict):
    if fout_test and random.random() < test_ratio:
      fout_test.write(json.dumps(rec, ensure_ascii=False) + '\n')
    else:
      fout_train.write(json.dumps(rec, ensure_ascii=False) + '\n')

  for ex in tqdm(ultra_ds, desc='UltraChat 200k training'):
    fout_train.write(json.dumps(_convert_ultrachat(ex), ensure_ascii=False) + '\n')
  for ext in tqdm(ultra_test, desc='UltraChat 200k test'):
    fout_test.write(json.dumps(_convert_ultrachat(ext), ensure_ascii=False) + '\n')

  for ex in tqdm(share_ds, desc='ShareGPT'):
    _write(_convert_sharegpt(ex))

  fout_train.close()
  if fout_test:
    fout_test.close()

  msg = f'\n✅ Mixed dataset written to {train_path.resolve()}'
  if test_path is not None and test_ratio > 0.0:
    msg += f' (test subset: {test_path.resolve()}, ratio={test_ratio})'
  print(msg)


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #


def cli() -> None:
  parser = argparse.ArgumentParser(description='Create the mixed EAGLE-3 training dataset')
  parser.add_argument(
    '--out',
    type=str,
    default='eagle/data/eagle3_mixed_ultra_sharegpt.jsonl',
    help='Destination JSONL file for the training split (default: %(default)s)',
  )
  parser.add_argument(
    '--test-out', type=str, default=None, help='Optional path to write a held-out test split (disabled if omitted).'
  )
  parser.add_argument(
    '--test-ratio',
    type=float,
    default=0.02,
    help='Fraction of examples to allocate to the test split (0-1). Only takes effect if --test-out is provided.',
  )
  parser.add_argument('--overwrite', action='store_true', help='Overwrite the output file if it already exists')
  args = parser.parse_args()

  build_dataset(
    Path(args.out),
    overwrite=args.overwrite,
    test_path=Path(args.test_out) if args.test_out else None,
    test_ratio=args.test_ratio if args.test_out else 0.0,
  )


if __name__ == '__main__':
  cli()
