#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=wait_for_everyone_hang
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --output=../audiolmWatermark-pytorch-results/output-%A.log
#SBATCH --error=../audiolmWatermark-pytorch-results/error-%A.log
#SBATCH --requeue

accelerate launch wait_for_everyone_hang.py