#!/bin/bash

# A script for a continuous evaluation of the provided list of models.
#
# Usage:
# bash eval.sh <comma-separated-list-of-models> <gpu-id>
#
# Example:
# bash eval.sh tless-bop20-xc65-f64,ycbv-bop20-xc65-f64 0


# Exit immediately if a command exits with a non-zero status.
set -e

# Parse arguments.
if [ $# -ne 2 ]
  then
    echo "arguments: models gpu_id"
    exit 1
fi
MODELS=$1  # Comma-separated list of model names.
GPU=$2

echo "Models to evaluate: $MODELS"
echo "GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

# Parse the list of model names.
MODELS_ARR=(${MODELS//,/ })

while :
do
    for MODEL in "${MODELS_ARR[@]}"
    do
        echo "#################################################################"
        echo "EVALUATING: $MODEL"
        echo "#################################################################"

        MODEL_DIR="$TF_MODELS_PATH/$MODEL"
        echo "Model directory: $MODEL_DIR"

        python eval.py --model="$MODEL" |& ./tee.py "$MODEL_DIR/log_eval.txt"
    done
done
