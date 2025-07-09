#!/usr/bin/env python3
"""
Example usage script for EAGLE-3 training with mixed datasets.
This script demonstrates different ways to use the updated training code.
"""

import subprocess
import os
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr[-500:])  # Show last 500 chars
        return False


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'datasets', 'transformers', 'torch', 'deepspeed', 
        'accelerate', 'wandb', 'tqdm'
    ]
    
    print("Checking dependencies...")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✓ All dependencies satisfied")
    return True


def download_datasets_only():
    """Download and process datasets without training"""
    print("Downloading and processing datasets...")
    
    # Create a simple script to just download datasets
    script_content = """
import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import download_and_mix_datasets

if __name__ == '__main__':
    print("Downloading and mixing UltraChat_200k and ShareGPT datasets...")
    train_path, test_path = download_and_mix_datasets(cache_dir="./data_cache")
    print(f"✓ Training data: {train_path}")
    print(f"✓ Test data: {test_path}")
    print("Datasets are ready for training!")
"""
    
    with open("download_datasets.py", "w") as f:
        f.write(script_content)
    
    return run_command(
        ["python", "download_datasets.py"],
        "Download and process datasets"
    )


def example_basic_training():
    """Example: Basic training with default settings"""
    cmd = [
        "deepspeed", "main.py",
        "--deepspeed_config", "ds_config.json"
    ]
    
    return run_command(cmd, "Basic EAGLE-3 training with mixed datasets")


def example_custom_model():
    """Example: Training with a custom base model"""
    cmd = [
        "deepspeed", "main.py",
        "--deepspeed_config", "ds_config.json",
        "--basepath", "microsoft/DialoGPT-medium",  # Alternative base model
        "--cache_dir", "./custom_cache",
        "--savedir", "./custom_checkpoints"
    ]
    
    return run_command(cmd, "Training with custom base model")


def example_resume_training():
    """Example: Resume training from checkpoint"""
    cmd = [
        "deepspeed", "main.py",
        "--deepspeed_config", "ds_config.json",
        "--savedir", "./checkpoints"  # Will auto-detect latest checkpoint
    ]
    
    return run_command(cmd, "Resume training from latest checkpoint")


def example_custom_datasets():
    """Example: Training with custom datasets"""
    # First, let's create some example custom data
    example_train = [
        {
            "id": "custom_0",
            "conversations": [
                {"from": "human", "value": "What is machine learning?"},
                {"from": "gpt", "value": "Machine learning is a subset of artificial intelligence..."},
                {"from": "human", "value": "Can you give me an example?"},
                {"from": "gpt", "value": "Sure! An example of machine learning is..."}
            ]
        }
    ]
    
    import json
    os.makedirs("./custom_data", exist_ok=True)
    
    with open("./custom_data/train.jsonl", "w") as f:
        for item in example_train:
            f.write(json.dumps(item) + "\n")
    
    with open("./custom_data/test.jsonl", "w") as f:
        for item in example_train:  # Using same data for simplicity
            f.write(json.dumps(item) + "\n")
    
    cmd = [
        "deepspeed", "main.py",
        "--deepspeed_config", "ds_config.json",
        "--trainpath", "./custom_data/train.jsonl",
        "--testpath", "./custom_data/test.jsonl",
        "--savedir", "./custom_data_checkpoints"
    ]
    
    return run_command(cmd, "Training with custom datasets")


def create_example_config():
    """Create an example DeepSpeed configuration"""
    config = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 8,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 5e-5,
                "warmup_num_steps": 1000
            }
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "wall_clock_breakdown": False
    }
    
    import json
    with open("example_ds_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created example_ds_config.json")


def main():
    parser = argparse.ArgumentParser(description="EAGLE-3 Training Examples")
    parser.add_argument("--example", choices=[
        "check", "download", "basic", "custom-model", 
        "resume", "custom-data", "config"
    ], default="check", help="Example to run")
    
    args = parser.parse_args()
    
    if args.example == "check":
        if not check_dependencies():
            print("\nPlease install missing dependencies before training.")
            return
        print("\n✓ Ready to train! Try other examples:")
        print("  python example_usage.py --example download")
        print("  python example_usage.py --example basic")
        
    elif args.example == "download":
        download_datasets_only()
        
    elif args.example == "basic":
        if not os.path.exists("ds_config.json"):
            print("Creating example DeepSpeed config...")
            create_example_config()
            print("Use example_ds_config.json as your ds_config.json")
        example_basic_training()
        
    elif args.example == "custom-model":
        example_custom_model()
        
    elif args.example == "resume":
        example_resume_training()
        
    elif args.example == "custom-data":
        example_custom_datasets()
        
    elif args.example == "config":
        create_example_config()


if __name__ == "__main__":
    main()