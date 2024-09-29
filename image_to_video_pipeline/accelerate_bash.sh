#!/bin/bash
#SBATCH --job-name=judge-each-video
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err
#SBATCH --exclusive

# 定义默认参数值
OUTPUT_JSON="output_files/HDPv2.json"
VIDEO_OUTPUT_DIR="videos"
CACHE_DIR="stable_video_diffusion"
PROCESS_DATASET_SCRIPT="process_dataset/hdpv2_process.py"
SAMPLE_SIZE=10

# 定义超参数
FRAME_COUNT=16
FRAME_DURATION=100

# 从命令行获取 index 和 split，允许默认值
INDEX=${1:-0}  # 如果不传入参数，默认 index 为 0
SPLIT=${2:-"test"}  # 如果不传入参数，默认 split 为 "test"

echo "Processing index: $INDEX"
echo "Processing split: $SPLIT"

# 执行 Python 脚本，并将 index 和 split 作为参数传递
accelerate launch accelerate_parallel_generate.py \
    --dataset_loader $PROCESS_DATASET_SCRIPT \
    --output_path $OUTPUT_JSON \
    --video_path $VIDEO_OUTPUT_DIR \
    --frame $FRAME_COUNT \
    --duration $FRAME_DURATION \
    --cache_dir $CACHE_DIR \
    --use_bfloat16 \
    --percentage 1 \
    --start_index $INDEX \
    --split $SPLIT

