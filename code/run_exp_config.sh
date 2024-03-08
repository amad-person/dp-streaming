#!/bin/bash

CONFIG="../configs/ny_taxi_exp_config.yaml"

python3 -m cProfile -o naive_nyc.pstats run_naive.py $CONFIG &
python3 -m cProfile -o binary_nyc.pstats run_binary.py $CONFIG &
python3 -m cProfile -o interval_nyc.pstats run_interval.py $CONFIG &

wait

echo "All mechanisms finished running!"