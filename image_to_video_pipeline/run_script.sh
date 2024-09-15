#!/bin/bash

# 定义各个路径
INPUT_JSON="/remote_shome/snl/feilong/xiapeng/haibo/image_to_video_pipeline/input_files/examples.json"
OUTPUT_JSON="/remote_shome/snl/feilong/xiapeng/haibo/image_to_video_pipeline/output_files/examples.json"
VIDEO_OUTPUT_DIR="/remote_shome/snl/feilong/xiapeng/haibo/image_to_video_pipeline/videos"
CACHE_DIR="/remote_shome/snl/feilong/xiapeng/haibo/videoRM/Stable_Video_Diffusion"

# 定义超参数
FRAME_COUNT=16
FRAME_DURATION=100

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 打印 CUDA 设备信息，供调试使用
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

# 执行 Python 脚本
python generate.py \
    --input_path $INPUT_JSON \
    --output_path $OUTPUT_JSON \
    --video_path $VIDEO_OUTPUT_DIR \
    --frame $FRAME_COUNT \
    --duration $FRAME_DURATION \
    --cache_dir $CACHE_DIR \
    --debug

echo "Script completed, output generated at $OUTPUT_JSON"
