#!/bin/bash
#SBATCH --partition h100
#SBATCH --account=vita
#SBATCH --chdir /work/vita/nie/haibo/image_to_video_pipeline
#SBATCH --job-name=hdpv2_video_generation      # 任务名称
#SBATCH --output=slurm_output_%j.out     # 标准输出日志文件（%j将被任务ID替换）
#SBATCH --error=slurm_error_%j.err       # 标准错误日志文件                    # 使用的任务数（一般为1）            # 每个任务使用的CPU核心数
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1                       # 分配的GPU数量
#SBATCH --mem=90G                        # 分配的内存大小                 # 使用的分区，假设你有gpu分区
#SBATCH --time=24:00:00                  # 最大运行时间，格式为 hh:mm:ss

module load gcc cuda/12.4.1
source ~/miniconda3/bin/activate video

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
python accelerate launch accelerate_parallel_generate.py \
    --dataset_loader $PROCESS_DATASET_SCRIPT \
    --output_path $OUTPUT_JSON \
    --video_path $VIDEO_OUTPUT_DIR \
    --frame $FRAME_COUNT \
    --duration $FRAME_DURATION \
    --cache_dir $CACHE_DIR \
    --use_bfloat16

