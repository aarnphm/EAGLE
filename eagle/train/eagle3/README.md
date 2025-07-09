# EAGLE-3 Training with Mixed Datasets

This directory contains the training code for EAGLE-3 that has been updated to use publicly available datasets instead of hardcoded local paths.

## Dataset Sources

The training script now automatically downloads and mixes data from:

1. **UltraChat_200k** (`HuggingFaceH4/ultrachat_200k`): 
   - High-quality multi-turn conversations
   - ~200k conversation examples
   - Uses the `train_sft` split

2. **ShareGPT** (`anon8231489123/ShareGPT_Vicuna_unfiltered`):
   - Community-generated conversations 
   - Filtered and cleaned dataset
   - Falls back to `RyokoAI/ShareGPT52K` if primary source unavailable

## Key Features

- **Automatic Dataset Download**: No need for manual dataset preparation
- **Format Unification**: Automatically converts different conversation formats to a unified structure
- **Robust Fallbacks**: Multiple ShareGPT sources to ensure availability
- **Configurable Caching**: Downloaded datasets are cached locally to avoid re-downloading
- **Mixed Training**: Combines diverse conversation styles from both datasets

## Usage

### Basic Training

```bash
cd eagle/train/eagle3
deepspeed main.py --deepspeed_config ds_config.json
```

### Custom Configuration

```bash
# Specify custom base model
deepspeed main.py \
    --deepspeed_config ds_config.json \
    --basepath meta-llama/Llama-3.1-8B-Instruct \
    --cache_dir ./my_data_cache \
    --savedir ./my_checkpoints

# Use your own datasets (optional)
deepspeed main.py \
    --deepspeed_config ds_config.json \
    --trainpath /path/to/your/train.jsonl \
    --testpath /path/to/your/test.jsonl
```

### Available Arguments

- `--basepath`: HuggingFace model name or local path (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `--cache_dir`: Directory to cache downloaded datasets (default: `./data_cache`)
- `--trainpath`: Custom training data path (optional - will auto-generate if not provided)
- `--testpath`: Custom test data path (optional - will auto-generate if not provided)
- `--savedir`: Directory to save checkpoints (default: `./checkpoints`)

## Dataset Format

The script converts all datasets to a unified JSONL format:

```json
{
  "id": "ultrachat_0",
  "conversations": [
    {"from": "human", "value": "User message here"},
    {"from": "gpt", "value": "Assistant response here"},
    {"from": "human", "value": "Follow-up message"},
    {"from": "gpt", "value": "Another response"}
  ]
}
```

## Data Processing Pipeline

1. **Download**: Fetches UltraChat_200k and ShareGPT datasets from HuggingFace Hub
2. **Format Conversion**: 
   - UltraChat: `{'role': 'user/assistant', 'content': '...'}` → `{'from': 'human/gpt', 'value': '...'}`
   - ShareGPT: Various formats → unified format
3. **Quality Filtering**: Removes conversations with less than 2 exchanges
4. **Mixing**: Combines datasets and shuffles
5. **Splitting**: Creates 95% train / 5% test split
6. **Caching**: Saves processed data as JSONL files for future use

## Hardware Requirements

- **GPU Memory**: 16GB+ recommended for training
- **Storage**: 10GB+ for dataset cache
- **RAM**: 32GB+ recommended

## Dependencies

Ensure you have the required packages:

```bash
pip install datasets transformers torch deepspeed accelerate wandb tqdm
```

## Troubleshooting

### Dataset Download Issues

If you encounter download errors:

1. Check your internet connection
2. Try clearing the cache: `rm -rf ./data_cache`
3. Use a different cache directory: `--cache_dir /tmp/eagle_cache`

### Memory Issues

If you run out of memory:

1. Reduce batch size in `ds_config.json`
2. Use gradient checkpointing
3. Consider using a smaller base model

### Format Errors

If you see conversation format errors:

1. Check that your custom datasets follow the expected JSONL format
2. Ensure conversations have alternating human/gpt messages
3. Verify that message content is not empty

## Performance Notes

- **First Run**: Will download ~2-3GB of datasets (cached for future runs)
- **Processing Time**: Initial dataset processing takes 10-20 minutes
- **Training Speed**: Depends on hardware; typically 1-2 hours per epoch on modern GPUs

## Monitoring

The script logs to Weights & Biases (wandb). Make sure to:

1. Set your wandb API key: `wandb login`
2. Configure the project name in the script if desired

## Contributing

When adding new dataset sources:

1. Add them to the `download_and_mix_datasets()` function
2. Implement format conversion to the unified structure
3. Test with a small subset first
4. Update this README with the new source