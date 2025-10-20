#!/bin/bash

SEED=${1:-0}
GPU_IDS=($2)

NUM_GPUS=${#GPU_IDS[@]}
if [ "$NUM_GPUS" -gt 8 ]; then
  echo "Error: Too many GPUs specified ($NUM_GPUS). Max allowed is 8."
  exit 1
fi

echo "Launching training with seed=$SEED on GPUs: ${GPU_IDS[*]}"

for GPU_ID in "${GPU_IDS[@]}"; do
  SCREEN_NAME="dynaformer_train_${GPU_ID}"
  LOG_FILE="train${GPU_ID}.log"
  CONFIG_FILE="configs${GPU_ID}.yaml"
  COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --mode train --config $CONFIG_FILE --seed $SEED --split protein_seqid > $LOG_FILE 2>&1; exit"

  echo "Starting training on GPU $GPU_ID (screen: $SCREEN_NAME)..."

  screen -dmS "$SCREEN_NAME" bash -c "$COMMAND"
  sleep 10  # Stagger the startups slightly
done

echo "All trainings launched successfully!"
