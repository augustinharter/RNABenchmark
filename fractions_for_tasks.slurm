#!/bin/bash
#SBATCH --job-name=fractions_for_tasks     # create a short name for your job

# Ressourcen pro einzelnem Experiment (Task):
#SBATCH --nodes=1              # Jeder Task braucht nur einen Node
#SBATCH --ntasks=1             # Ein Task pro Array-Index
#SBATCH --gres=gpu:1           # Reserviert genau eine GPU pro Task
#SBATCH --cpus-per-task=4      # CPU-Kerne pro GPU (je nach System anpassen)
#SBATCH --mem=16G              # RAM pro GPU-Task
#SBATCH --time=1-00:00:00      # Maximal 1 Tag Laufzeit pro Task

# %16 sorgt dafür, dass nie mehr als 16 Jobs gleichzeitig laufen
#SBATCH --array=0-7%16

# Fractions
fracs=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)
echo "Running with fraction: ${fracs[$SLURM_ARRAY_TASK_ID]} and task id $SLURM_ARRAY_TASK_ID"
SELECTED_TASK=SpliceAI START_FRACTION=${fracs[$SLURM_ARRAY_TASK_ID]} srun -o "out/fractions/%A/splice/${fracs[$SLURM_ARRAY_TASK_ID]}" -e "out/fractions/%A/splice/${fracs[$SLURM_ARRAY_TASK_ID]}" bash ./scripts/BEACON-B/all_task.sh