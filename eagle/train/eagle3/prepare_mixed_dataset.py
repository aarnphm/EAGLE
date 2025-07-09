#!/usr/bin/env python3
"""
Dataset preparation script for EAGLE3 training.
Mixes UltraChat_200k and ShareGPT datasets into a unified format.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from tqdm import tqdm


def convert_ultrachat_to_standard_format(conversation: List[str]) -> List[Dict[str, str]]:
    """Convert UltraChat format to standard conversation format."""
    conversations = []
    for i in range(0, len(conversation), 2):
        if i < len(conversation):
            conversations.append({
                "from": "human",
                "value": conversation[i]
            })
        if i + 1 < len(conversation):
            conversations.append({
                "from": "gpt", 
                "value": conversation[i + 1]
            })
    return conversations


def convert_sharegpt_to_standard_format(conversation: List[Dict]) -> List[Dict[str, str]]:
    """Convert ShareGPT format to standard conversation format."""
    conversations = []
    for turn in conversation:
        if turn["from"] in ["human", "user"]:
            conversations.append({
                "from": "human",
                "value": turn["value"]
            })
        elif turn["from"] in ["gpt", "assistant", "chatgpt"]:
            conversations.append({
                "from": "gpt",
                "value": turn["value"]
            })
    return conversations


def load_ultrachat_200k(max_samples: Optional[int] = None) -> List[Dict]:
    """Load UltraChat 200k dataset."""
    print("Loading UltraChat_200k dataset...")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    
    processed_data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing UltraChat")):
        if max_samples and idx >= max_samples:
            break
            
        # UltraChat format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        conversations = []
        for msg in item["messages"]:
            if msg["role"] == "user":
                conversations.append({
                    "from": "human",
                    "value": msg["content"]
                })
            elif msg["role"] == "assistant":
                conversations.append({
                    "from": "gpt",
                    "value": msg["content"]
                })
        
        if conversations:
            processed_data.append({
                "id": f"ultrachat_{idx}",
                "conversations": conversations
            })
    
    return processed_data


def load_sharegpt(max_samples: Optional[int] = None) -> List[Dict]:
    """Load ShareGPT dataset."""
    print("Loading ShareGPT dataset...")
    
    # Try to load ShareGPT from multiple sources
    try:
        # Try loading from the anon8231489123/ShareGPT_Vicuna_unfiltered dataset
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
    except:
        try:
            # Alternative: RyokoAI/ShareGPT52K
            dataset = load_dataset("RyokoAI/ShareGPT52K", split="train")
        except:
            print("Warning: Could not load ShareGPT dataset from HuggingFace. Using empty dataset.")
            return []
    
    processed_data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing ShareGPT")):
        if max_samples and idx >= max_samples:
            break
        
        # ShareGPT format varies, but typically has "conversations" field
        conversations = []
        if "conversations" in item:
            for turn in item["conversations"]:
                if turn["from"] in ["human", "user"]:
                    conversations.append({
                        "from": "human",
                        "value": turn["value"]
                    })
                elif turn["from"] in ["gpt", "assistant", "chatgpt"]:
                    conversations.append({
                        "from": "gpt",
                        "value": turn["value"]
                    })
        
        if conversations:
            processed_data.append({
                "id": f"sharegpt_{idx}",
                "conversations": conversations
            })
    
    return processed_data


def mix_datasets(ultrachat_data: List[Dict], sharegpt_data: List[Dict], 
                 mix_ratio: float = 0.5, seed: int = 42) -> List[Dict]:
    """Mix two datasets according to the specified ratio."""
    random.seed(seed)
    
    # Calculate how many samples from each dataset
    total_samples = len(ultrachat_data) + len(sharegpt_data)
    ultrachat_samples = int(total_samples * mix_ratio)
    sharegpt_samples = total_samples - ultrachat_samples
    
    # Sample from each dataset
    if ultrachat_samples > len(ultrachat_data):
        ultrachat_samples = len(ultrachat_data)
        sharegpt_samples = total_samples - ultrachat_samples
    if sharegpt_samples > len(sharegpt_data):
        sharegpt_samples = len(sharegpt_data)
        ultrachat_samples = total_samples - sharegpt_samples
    
    sampled_ultrachat = random.sample(ultrachat_data, min(ultrachat_samples, len(ultrachat_data)))
    sampled_sharegpt = random.sample(sharegpt_data, min(sharegpt_samples, len(sharegpt_data)))
    
    # Combine and shuffle
    mixed_data = sampled_ultrachat + sampled_sharegpt
    random.shuffle(mixed_data)
    
    print(f"Mixed dataset contains {len(sampled_ultrachat)} UltraChat samples and {len(sampled_sharegpt)} ShareGPT samples")
    
    return mixed_data


def save_dataset(data: List[Dict], output_path: str, split_ratio: float = 0.95):
    """Save dataset to JSON files with train/test split."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle data
    random.shuffle(data)
    
    # Split into train and test
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Save train data
    train_path = output_path
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(train_data)} training samples to {train_path}")
    
    # Save test data
    test_path = train_path.replace('.jsonl', '_test.jsonl').replace('.json', '_test.json')
    with open(test_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(test_data)} test samples to {test_path}")
    
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed dataset for EAGLE3 training")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./data/eagle3_mixed",
        help="Output directory for the mixed dataset"
    )
    parser.add_argument(
        "--ultrachat-max-samples", 
        type=int, 
        default=100000,
        help="Maximum number of samples to use from UltraChat (default: 100000)"
    )
    parser.add_argument(
        "--sharegpt-max-samples", 
        type=int, 
        default=100000,
        help="Maximum number of samples to use from ShareGPT (default: 100000)"
    )
    parser.add_argument(
        "--mix-ratio", 
        type=float, 
        default=0.5,
        help="Ratio of UltraChat samples in the final mix (default: 0.5)"
    )
    parser.add_argument(
        "--split-ratio", 
        type=float, 
        default=0.95,
        help="Train/test split ratio (default: 0.95)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    ultrachat_data = load_ultrachat_200k(args.ultrachat_max_samples)
    sharegpt_data = load_sharegpt(args.sharegpt_max_samples)
    
    # Mix datasets
    mixed_data = mix_datasets(
        ultrachat_data, 
        sharegpt_data, 
        mix_ratio=args.mix_ratio,
        seed=args.seed
    )
    
    # Save mixed dataset
    output_path = output_dir / "mixed_dataset.jsonl"
    train_path, test_path = save_dataset(mixed_data, str(output_path), args.split_ratio)
    
    # Save metadata
    metadata = {
        "ultrachat_samples": len(ultrachat_data),
        "sharegpt_samples": len(sharegpt_data),
        "total_samples": len(mixed_data),
        "mix_ratio": args.mix_ratio,
        "split_ratio": args.split_ratio,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "seed": args.seed
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")


if __name__ == "__main__":
    main()