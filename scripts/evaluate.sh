#!/bin/bash

# Simple script to evaluate a model on a VTAB dataset test set.
# This follows the paper's evaluation protocol: training on the combined
# train+val split (800+200=1000 samples) and evaluating on the test split.

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <adapter> <backbone> <dataset> [additional_hydra_args...]"
    echo "Example: $0 combo in21k vtab_cifar100 seed=42"
    exit 1
fi

ADAPTER=$1
BACKBONE=$2
DATASET=$3
shift 3
EXTRA_ARGS="$@"

echo "========================================================="
echo "Evaluating ${ADAPTER} with ${BACKBONE} on ${DATASET}"
echo "Protocol: Train on 1000 samples (train+val), evaluate on test"
echo "========================================================="

# The key overrides for test-set evaluation:
# 1. Use the combined train800+val200 split for training
# 2. Set test_after_training=true to run the test set after training
# 3. We disable validation checks during training to save time since 
#    we are just training for a fixed number of steps (1600, scaled from 1300 
#    to account for the 1000/800 sample ratio and maintain the same epochs).

python train.py \
    adapter=${ADAPTER} \
    backbone=${BACKBONE} \
    data=${DATASET} \
    data.train_split_name="train800val200" \
    trainer.test_after_training=true \
    trainer.limit_val_batches=0 \
    training.max_steps=1600 \
    ${EXTRA_ARGS}

echo "Evaluation complete. See WandB (if enabled) or logs for test_acc metric."
