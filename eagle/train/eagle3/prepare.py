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

import argparse, json, uuid, random

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ---------------------------------------------------------------------------- #
# Conversation utilities
# ---------------------------------------------------------------------------- #


def _enforce_turn_order(messages: list[dict]) -> list[dict]:
  """Return a copy of *messages* conformed to `[system]? human gpt human gpt ...`.

  Rules
  -----
  1. At most one leading *system* message is kept (if multiple provided,
     only the first is retained; the rest are discarded).
  2. After the optional *system*, turns must strictly alternate between
     *human* and *gpt* – starting with *human*.
  3. Any message breaking the pattern is skipped.
  4. If the conversation ends with an unmatched *human* turn (i.e. no
     following *gpt*), that final message is dropped so that every *human*
     query has a corresponding assistant response.
  """

  if not messages:
    return []

  cleaned = []

  # Handle (optional) initial system message(s)
  itr = iter(messages)
  first_msg = next(itr, None)
  if first_msg and first_msg['from'] == 'system':
    cleaned.append(first_msg)
    # consume any extra leading system messages silently
    for m in itr:
      if m['from'] != 'system':
        # rewind the iterator one step back via a list trick
        remaining = [m] + list(itr)
        itr = iter(remaining)
        break

  # Now enforce alternating human → gpt pattern
  expect = 'human'
  for msg in itr:
    role = msg.get('from')
    if role != expect:
      continue  # skip messages out of order / unwanted roles
    cleaned.append(msg)
    expect = 'gpt' if expect == 'human' else 'human'

  # Ensure conversation ends with assistant (gpt). If odd length after system
  # removal, drop trailing human.
  if cleaned and cleaned[-1]['from'] == 'human':
    cleaned.pop()

  return cleaned


# ---------------------------------------------------------------------------- #
# Conversion helpers
# ---------------------------------------------------------------------------- #


def _convert_ultrachat(example: dict) -> dict:
  """Convert a single UltraChat example into ShareGPT format."""
  messages = []
  role_map = {'user': 'human', 'assistant': 'gpt', 'system': 'system'}

  for msg in example.get('messages', []):
    mapped = role_map.get(msg.get('role'))
    if mapped is None:
      continue
    messages.append({'from': mapped, 'value': msg.get('content', '')})

  messages = _enforce_turn_order(messages)
  if len(messages) < 2:  # need at least one human–gpt pair
    return {}

  return {'id': str(uuid.uuid4()), 'conversations': messages}


def _convert_sharegpt(example) -> dict:
  """Normalise ShareGPT roles to `human`, `gpt`, or `system` and drop others."""

  role_map = {
    'human': 'human',
    'user': 'human',
    'assistant': 'gpt',
    'gpt': 'gpt',
    'chatgpt': 'gpt',
    'system': 'system',
  }

  conversations = []
  for c in example['conversations']:
    mapped = role_map.get(c.get('from'))
    if mapped is None:
      continue
    conversations.append({'from': mapped, 'value': c['value']})

  conversations = _enforce_turn_order(conversations)
  if len(conversations) < 2:
    return {}

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

  for ex in tqdm(ultra_ds, desc='UltraChat 200k [train]'):
    fout_train.write(json.dumps(_convert_ultrachat(ex), ensure_ascii=False) + '\n')
  for ext in tqdm(ultra_test, desc='UltraChat 200k [test]'):
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
