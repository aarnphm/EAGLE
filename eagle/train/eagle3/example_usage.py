#!/usr/bin/env python3
"""
Example usage of the EAGLE3 mixed dataset training pipeline.

This script demonstrates different ways to use the dataset preparation
and training functionality.
"""

import subprocess
import os
import sys


def run_command(cmd):
    """Run a shell command and print the output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    return result


def example_1_basic_training():
    """Example 1: Basic training with default settings."""
    print("\n=== Example 1: Basic Training ===")
    print("This will prepare a mixed dataset and start training with default settings.")
    
    # Run the training script with defaults
    run_command("./run_training.sh")


def example_2_custom_mix_ratio():
    """Example 2: Training with custom dataset mix ratio."""
    print("\n=== Example 2: Custom Mix Ratio ===")
    print("This example uses 70% UltraChat and 30% ShareGPT data.")
    
    # Prepare dataset with custom mix ratio
    run_command("""
        python prepare_mixed_dataset.py \
            --output-dir ./data/eagle3_custom_mix \
            --ultrachat-max-samples 70000 \
            --sharegpt-max-samples 30000 \
            --mix-ratio 0.7
    """)
    
    # Train with the custom dataset
    run_command("""
        deepspeed main.py \
            --deepspeed ds_config.json \
            --basepath meta-llama/Llama-3.1-8B-Instruct \
            --trainpath ./data/eagle3_custom_mix/mixed_dataset.jsonl \
            --testpath ./data/eagle3_custom_mix/mixed_dataset_test.jsonl \
            --savedir ./checkpoints/eagle3_custom_mix
    """)


def example_3_small_dataset_test():
    """Example 3: Quick test with small dataset."""
    print("\n=== Example 3: Small Dataset Test ===")
    print("This example uses a small dataset for quick testing.")
    
    # Prepare small dataset
    run_command("""
        python prepare_mixed_dataset.py \
            --output-dir ./data/eagle3_test \
            --ultrachat-max-samples 1000 \
            --sharegpt-max-samples 1000 \
            --mix-ratio 0.5
    """)
    
    # You can then train with this small dataset
    print("\nTo train with this small dataset, run:")
    print("""
        deepspeed main.py \
            --deepspeed ds_config.json \
            --basepath meta-llama/Llama-3.1-8B-Instruct \
            --trainpath ./data/eagle3_test/mixed_dataset.jsonl \
            --testpath ./data/eagle3_test/mixed_dataset_test.jsonl \
            --savedir ./checkpoints/eagle3_test
    """)


def example_4_dataset_only():
    """Example 4: Prepare dataset only (no training)."""
    print("\n=== Example 4: Dataset Preparation Only ===")
    print("This example only prepares the dataset without starting training.")
    
    run_command("""
        python prepare_mixed_dataset.py \
            --output-dir ./data/eagle3_prepared \
            --ultrachat-max-samples 50000 \
            --sharegpt-max-samples 50000 \
            --mix-ratio 0.5 \
            --split-ratio 0.9
    """)
    
    print("\nDataset prepared! You can now use it for training later.")


def print_usage():
    """Print usage information."""
    print("""
EAGLE3 Mixed Dataset Training Examples

This script provides examples of how to use the EAGLE3 training pipeline
with mixed UltraChat and ShareGPT datasets.

Usage:
    python example_usage.py [example_number]

Examples:
    1 - Basic training with default settings
    2 - Training with custom dataset mix ratio
    3 - Quick test with small dataset
    4 - Dataset preparation only (no training)

If no example number is provided, all examples will be shown (without running).
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == "1":
            example_1_basic_training()
        elif example_num == "2":
            example_2_custom_mix_ratio()
        elif example_num == "3":
            example_3_small_dataset_test()
        elif example_num == "4":
            example_4_dataset_only()
        else:
            print(f"Unknown example number: {example_num}")
            print_usage()
    else:
        print_usage()
        print("\n=== Available Examples (not running) ===")
        print("\n1. Basic Training:")
        print("   ./run_training.sh")
        
        print("\n2. Custom Mix Ratio (70% UltraChat, 30% ShareGPT):")
        print("   ./run_training.sh --mix-ratio 0.7")
        
        print("\n3. Small Dataset Test:")
        print("   ./run_training.sh --ultrachat-samples 1000 --sharegpt-samples 1000")
        
        print("\n4. Dataset Preparation Only:")
        print("   python prepare_mixed_dataset.py --output-dir ./data/eagle3_mixed")