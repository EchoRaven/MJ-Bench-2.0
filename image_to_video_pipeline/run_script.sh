#!/bin/bash

# 定义各个路径
OUTPUT_JSON="output_files/HDPv2.json"
VIDEO_OUTPUT_DIR="videos"
CACHE_DIR="stable_video_diffusion"
PROCESS_DATASET_SCRIPT="process_dataset/hdpv2_process.py"
SAMPLE_SIZE=10

# 定义超参数
FRAME_COUNT=16
FRAME_DURATION=100

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 打印 CUDA 设备信息，供调试使用
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

# 使用 nohup 执行 Python 脚本并将输出重定向到 nohup.out
nohup python generate.py \
    --dataset_loader $PROCESS_DATASET_SCRIPT \
    --output_path $OUTPUT_JSON \
    --video_path $VIDEO_OUTPUT_DIR \
    --frame $FRAME_COUNT \
    --duration $FRAME_DURATION \
    --cache_dir $CACHE_DIR > nohup.out 2>&1 &

echo "Script running in the background, output logged in nohup.out"
