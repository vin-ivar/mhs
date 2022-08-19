#!/bin/bash
#SBATCH --account=ec30
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=4G
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error
source ~/.bashrc
conda activate $PROJ/envs/nlp
module --quiet purge  # Reset the modules to the system default

python -u mbert_multi.py
