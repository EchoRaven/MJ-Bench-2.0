#!/bin/bash

#SBATCH --job-name=video_generation      # 任务名称
#SBATCH --output=slurm_output_%j.out     # 标准输出日志文件（%j将被任务ID替换）
#SBATCH --error=slurm_error_%j.err       # 标准错误日志文件
#SBATCH --ntasks=1                       # 使用的任务数（一般为1）
#SBATCH --cpus-per-task=4                # 每个任务使用的CPU核心数
#SBATCH --gpus=4                         # 分配的GPU数量
#SBATCH --mem=32G                        # 分配的内存大小
#SBATCH --partition=gpu                  # 使用的分区，假设你有gpu分区
#SBATCH --time=24:00:00                  # 最大运行时间，格式为 hh:mm:ss

# 定义各个路径
OUTPUT_JSON="output_files/HDPv2.json"
VIDEO_OUTPUT_DIR="videos"
CACHE_DIR="stable_video_diffusion"
PROCESS_DATASET_SCRIPT="process_dataset/hdpv2_process.py"
SAMPLE_SIZE=10

# 定义超参数
FRAME_COUNT=16
FRAME_DURATION=100

# 执行 Python 脚本
python generate.py \
    --dataset_loader $PROCESS_DATASET_SCRIPT \
    --output_path $OUTPUT_JSON \
    --video_path $VIDEO_OUTPUT_DIR \
    --frame $FRAME_COUNT \
    --duration $FRAME_DURATION \
    --cache_dir $CACHE_DIR

