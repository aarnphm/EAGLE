# EAGLE3 Mixed Dataset Training

This directory contains scripts for training EAGLE3 models using a mixed dataset of UltraChat_200k and ShareGPT data. The setup has been designed to be portable and easy to use without hardcoded paths.

## Overview

The training pipeline now supports:
- Automatic downloading and mixing of UltraChat_200k and ShareGPT datasets
- Configurable mix ratios between the two datasets
- Portable paths that work across different environments
- Easy-to-use training script with sensible defaults

## Quick Start

### 1. Basic Usage

Run the training with default settings:

```bash
cd eagle/train/eagle3
./run_training.sh
```

This will:
- Download and mix UltraChat_200k and ShareGPT datasets (50/50 mix)
- Create train/test splits (95/5)
- Start training with the Llama-3.1-8B-Instruct base model

### 2. Custom Configuration

You can customize various aspects of the training:

```bash
# Use a different base model
./run_training.sh --base-model "meta-llama/Llama-2-7b-chat-hf"

# Change the mix ratio (0.7 = 70% UltraChat, 30% ShareGPT)
./run_training.sh --mix-ratio 0.7

# Use fewer samples for faster experimentation
./run_training.sh --ultrachat-samples 10000 --sharegpt-samples 10000

# Skip dataset preparation if you've already prepared the data
./run_training.sh --skip-data-prep
```

### 3. Manual Dataset Preparation

If you want to prepare the dataset separately:

```bash
python prepare_mixed_dataset.py \
  --output-dir ./data/eagle3_mixed \
  --ultrachat-max-samples 100000 \
  --sharegpt-max-samples 100000 \
  --mix-ratio 0.5
```

## Dataset Format

The mixed dataset follows the standard conversation format:

```json
{
  "id": "ultrachat_0",
  "conversations": [
    {"from": "human", "value": "Hello, how are you?"},
    {"from": "gpt", "value": "I'm doing well, thank you! How can I help you today?"}
  ]
}
```

## File Structure

After running the dataset preparation, you'll have:

```
data/eagle3_mixed/
├── mixed_dataset.jsonl         # Training data
├── mixed_dataset_test.jsonl    # Test data
└── dataset_metadata.json       # Metadata about the dataset
```

## Training Arguments

The main training script (`main.py`) now accepts:

- `--basepath`: Path to the base model (required)
- `--trainpath`: Path to training data (default: `./data/eagle3_mixed/mixed_dataset.jsonl`)
- `--testpath`: Path to test data (default: `./data/eagle3_mixed/mixed_dataset_test.jsonl`)
- `--savedir`: Directory for checkpoints (default: `./checkpoints/eagle3`)

## Using Custom Datasets

If you have your own dataset in the correct format, you can use it directly:

```bash
deepspeed main.py \
  --deepspeed ds_config.json \
  --basepath "your-base-model" \
  --trainpath "path/to/your/train.jsonl" \
  --testpath "path/to/your/test.jsonl" \
  --savedir "./checkpoints/custom"
```

## Troubleshooting

1. **Dataset download fails**: Make sure you have internet connection and the HuggingFace datasets library is properly installed.

2. **Out of memory**: Reduce the number of samples or adjust the batch size in `ds_config.json`.

3. **Permission errors**: Make sure the script has execute permissions: `chmod +x run_training.sh`

## Advanced Options

For more control, you can modify:

- `ds_config.json`: DeepSpeed configuration (batch size, learning rate, etc.)
- `config.json`: Model architecture configuration
- `prepare_mixed_dataset.py`: Dataset preparation logic

## Dataset Sources

- **UltraChat_200k**: Downloaded from `HuggingFaceH4/ultrachat_200k`
- **ShareGPT**: Downloaded from `anon8231489123/ShareGPT_Vicuna_unfiltered` or `RyokoAI/ShareGPT52K`

Both datasets are automatically downloaded and processed when you run the training script.