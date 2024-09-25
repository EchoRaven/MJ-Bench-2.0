#!/bin/bash
#SBATCH --job-name=judge-each-video
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err
#SBATCH --exclusive

accelerate launch label.py \
    --num_processes=8 \
    --json_file_path "../../videoRM/dataset/SafeSora/config-train.json" \
    --videos_dir "../../videoRM/dataset/SafeSora" \
    --model_path "OpenGVLab/InternVL2-26B" \
    --cache_dir "../../videoRM/Internvl/pretrained" \
    --alignment_results_file "./result/alignment_result.json" \
    --safety_results_file "./result/safety_result.json" \
    --bias_results_file "./result/bias_result.json" \
    --quality_results_file "./result/quality_result.json" \
    --cc_results_file "./result/cc_result.json"