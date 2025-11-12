#!/bin/bash

CHECKPOINT=$1
GPU_ID=${2:-0}  # Default GPU ID is 0 if not provided
# SCREEN_NAME="dynaformer_eval_${GPU_ID}"

COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --mode eval --checkpoint $CHECKPOINT; exit"

echo "Starting evaluating checkpoint $CHECKPOINT on GPU $GPU_ID..."
# screen -dmS "$SCREEN_NAME" bash -c "$COMMAND"

eval "$COMMAND"

# echo "Evaluation launched successfully!"
