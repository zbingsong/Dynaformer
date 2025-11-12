#!/bin/bash

SEED=${1:-0}  # Default seed is 0 if not provided
GPU_ID=${2:-0}  # Default GPU ID is 0 if not provided
SCREEN_NAME="dynaformer_${SEED}"
LOG_FILE="train_${SEED}.log"
CONFIG_FILE="configs.yaml"
COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --mode train --config $CONFIG_FILE --seed $SEED > $LOG_FILE 2>&1; exit"

echo "Starting training on GPU $GPU_ID, random seed $SEED (screen: $SCREEN_NAME)..."
screen -dmS "$SCREEN_NAME" bash -c "$COMMAND"

echo "Training launched successfully!"
