#!/bin/bash

BATCH_SIZE=5
WINDOW_SIZE=5

python3 ../data/preprocessor.py create_adult $BATCH_SIZE $WINDOW_SIZE

CONFIG="../configs/adult_ablation/adult_exp_config_b${BATCH_SIZE}_w${WINDOW_SIZE}.yaml"

python3 ../code/run_naive.py $CONFIG

echo "Naive Binary finished running!"