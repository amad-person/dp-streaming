#!/bin/bash

CONFIG="../configs/acs_public_cov_exp_config.yaml"

python3 abs_error_vs_batch.py --config=$CONFIG &
python3 ans_vs_batch.py --config=$CONFIG &
python3 rel_error_vs_batch.py --config=$CONFIG &
python3 runtime_vs_batch_per_task.py --config=$CONFIG &

wait

echo "All plots generated!"