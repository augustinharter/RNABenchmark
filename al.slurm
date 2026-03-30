#!/bin/bash
#SBATCH --job-name=spliceAI-AL     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:h100:1             # number of allocated gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --qos=normal

SELECTED_TASK=SpliceAI START_FRACTION=0.02 ITERATION_FRACTION=0.02 ITERATIONS=5 srun --ntasks=1 --exclusive --gres=gpu:1 bash ./scripts/BEACON-B/all_task.sh