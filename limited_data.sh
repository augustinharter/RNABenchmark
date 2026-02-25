#!/bin/bash

if [ $# -eq 0 ]; then
    fractions=(0.01 0.05 0.1 0.5)
else
    fractions=($@)
fi
# export GPU if not already set
if [ -z "$GPU" ]; then
    export GPU=1
fi
for frac in "${fractions[@]}"; do
    SELECTED_TASK=${SELECTED_TASK:-'all'}
    echo "Running $SELECTED_TASK task with SIZE_FRACTION=$frac"
    export SIZE_FRACTION=$frac
    bash scripts/BEACON-B/all_task.sh > logs/limited_data_${SELECTED_TASK}_${SIZE_FRACTION}.log
    # python rm_checkpoints.py ${SIZE_FRACTION}/${SELECTED_TASK}
    # python rm_checkpoints.py
done
