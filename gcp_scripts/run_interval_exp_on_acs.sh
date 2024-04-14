#!/bin/bash

BATCH_SIZE=10
WINDOW_SIZE=10

python3 ../data/preprocessor.py create_acs_public_cov $BATCH_SIZE $WINDOW_SIZE

CONFIG="../configs/acs_ablation/acs_public_cov_exp_config_b${BATCH_SIZE}_w${WINDOW_SIZE}.yaml"

python3 ../code/run_interval.py $CONFIG

echo "Interval Restarts finished running!"