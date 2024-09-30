#!/bin/bash
#SBATCH --job-name=moe
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-cpu=9200
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err
#SBATCH --exclusive

python eval_4.py