#!/bin/bash

if [ $# -eq 0 ]; then
    fractions=(0.01 0.05 0.1 0.5)
else
    fractions=($@)
fi
export GPU=1
for frac in "${fractions[@]}"; do
    echo "Running with SIZE_FRACTION=$frac"
    export SIZE_FRACTION=$frac
    bash scripts/BEACON-B/all_task.sh > logs/${frac}.txt
done
python rm_checkpoints.py