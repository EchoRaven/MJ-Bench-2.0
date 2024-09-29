#!/bin/bash
#SBATCH --job-name=judge-each-video
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err
#SBATCH --exclusive

accelerate launch caption.py \
    --json_file_path "hpdv2.json" \
    --videos_dir "../../image_to_vide" \
    --model_path "OpenGVLab/InternVL2-26B" \
    --cache_dir "../../videoRM/Internvl/pretrained" \
    --results_file "./result/hdpv2.json" \