#!/bin/bash
# Script to prepare dataset and run EAGLE3 training with mixed UltraChat and ShareGPT data

# Default values
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="./data/eagle3_mixed"
CHECKPOINT_DIR="./checkpoints/eagle3"
ULTRACHAT_SAMPLES=100000
SHAREGPT_SAMPLES=100000
MIX_RATIO=0.5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --ultrachat-samples)
      ULTRACHAT_SAMPLES="$2"
      shift 2
      ;;
    --sharegpt-samples)
      SHAREGPT_SAMPLES="$2"
      shift 2
      ;;
    --mix-ratio)
      MIX_RATIO="$2"
      shift 2
      ;;
    --skip-data-prep)
      SKIP_DATA_PREP=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --base-model MODEL         Base model path (default: meta-llama/Llama-3.1-8B-Instruct)"
      echo "  --data-dir DIR            Data directory (default: ./data/eagle3_mixed)"
      echo "  --checkpoint-dir DIR      Checkpoint directory (default: ./checkpoints/eagle3)"
      echo "  --ultrachat-samples N     Number of UltraChat samples (default: 100000)"
      echo "  --sharegpt-samples N      Number of ShareGPT samples (default: 100000)"
      echo "  --mix-ratio RATIO        Mix ratio for UltraChat (default: 0.5)"
      echo "  --skip-data-prep         Skip dataset preparation"
      echo "  --help                   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=== EAGLE3 Training Setup ==="
echo "Base Model: $BASE_MODEL"
echo "Data Directory: $DATA_DIR"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "UltraChat Samples: $ULTRACHAT_SAMPLES"
echo "ShareGPT Samples: $SHAREGPT_SAMPLES"
echo "Mix Ratio: $MIX_RATIO"
echo ""

# Step 1: Prepare the dataset if not skipped
if [ "$SKIP_DATA_PREP" != true ]; then
  echo "=== Step 1: Preparing Mixed Dataset ==="
  python prepare_mixed_dataset.py \
    --output-dir "$DATA_DIR" \
    --ultrachat-max-samples "$ULTRACHAT_SAMPLES" \
    --sharegpt-max-samples "$SHAREGPT_SAMPLES" \
    --mix-ratio "$MIX_RATIO"
  
  if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed!"
    exit 1
  fi
  echo "Dataset preparation completed successfully!"
  echo ""
else
  echo "=== Skipping dataset preparation ==="
  echo ""
fi

# Check if dataset exists
if [ ! -f "$DATA_DIR/mixed_dataset.jsonl" ]; then
  echo "Error: Training dataset not found at $DATA_DIR/mixed_dataset.jsonl"
  echo "Please run dataset preparation first (remove --skip-data-prep flag)"
  exit 1
fi

# Step 2: Run training
echo "=== Step 2: Starting EAGLE3 Training ==="
echo "Running DeepSpeed training..."

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Run the training
deepspeed main.py \
  --deepspeed ds_config.json \
  --basepath "$BASE_MODEL" \
  --trainpath "$DATA_DIR/mixed_dataset.jsonl" \
  --testpath "$DATA_DIR/mixed_dataset_test.jsonl" \
  --savedir "$CHECKPOINT_DIR"

if [ $? -eq 0 ]; then
  echo ""
  echo "=== Training completed successfully! ==="
  echo "Checkpoints saved to: $CHECKPOINT_DIR"
else
  echo ""
  echo "=== Training failed! ==="
  exit 1
fi