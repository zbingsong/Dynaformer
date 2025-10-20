#!/bin/bash

SEEDS=($1)
GPU_IDS=($2)

NUM_GPUS=${#GPU_IDS[@]}
if [ "$NUM_GPUS" -gt 8 ]; then
  echo "Error: Too many GPUs specified ($NUM_GPUS). Max allowed is 8."
  exit 1
fi

# echo "Launching training with seed=$SEED on GPUs: ${GPU_IDS[*]}"

for (( i=0; i<NUM_GPUS; i++ )); do
  SEED=${SEEDS[i]}
  GPU_ID=${GPU_IDS[i]}
  SCREEN_NAME="dynaformer_${GPU_ID}"
  LOG_FILE="train_${SEED}.log"
  CONFIG_FILE="configs.yaml"
  COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --mode train --config $CONFIG_FILE --seed $SEED > $LOG_FILE 2>&1; exit"

  echo "Starting training on GPU $GPU_ID, random seed $SEED (screen: $SCREEN_NAME)..."
  screen -dmS "$SCREEN_NAME" bash -c "$COMMAND"

  if [ $i -lt $((NUM_GPUS-1)) ]; then
    sleep 10  # Stagger the startups slightly
  fi
done

echo "All trainings launched successfully!"
