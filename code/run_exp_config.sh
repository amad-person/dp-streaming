#!/bin/bash

CONFIG="../configs/ny_taxi_exp_config.yaml"

python3 run_naive.py $CONFIG &
python3 run_binary.py $CONFIG &
python3 run_interval.py $CONFIG &

wait

echo "All mechanisms finished running!"