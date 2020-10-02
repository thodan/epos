#!/bin/bash

# A script to launch training of the specified model.
#
# Usage:
# bash train.sh <model> <gpu-id>
#
# Example:
# bash train.sh tless-bop20-xc65-f64 0


# Exit immediately if a command exits with a non-zero status.
set -e

# Parse arguments.
if [ $# -ne 2 ]
  then
    echo "arguments: model gpu_id"
    exit 1
fi
MODEL=$1
GPU=$2

MODEL_DIR="$TF_MODELS_PATH/$MODEL"
export CUDA_VISIBLE_DEVICES=$GPU

echo "Training: $MODEL, GPU: $GPU"
echo "Model directory: $MODEL_DIR"

python train.py --model="$MODEL" |& ./tee.py "$MODEL_DIR/log_train.txt"
