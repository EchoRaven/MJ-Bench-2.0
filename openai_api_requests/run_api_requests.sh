#!/bin/bash

# 检查输入参数
if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <input_file> <output_file> <config_file> <max_workers> <log_file> [--debug] [--sample_size <size>]"
  exit 1
fi

# 获取输入参数
INPUT_FILE=$1
OUTPUT_FILE=$2
CONFIG_FILE=$3
MAX_WORKERS=$4
LOG_FILE=$5

# 检查是否有调试模式参数
DEBUG_FLAG=""
SAMPLE_SIZE_FLAG=""

# 检查是否传入 --debug 参数
if [[ "$6" == "--debug" ]]; then
  DEBUG_FLAG="--debug"
  # 检查是否有 --sample_size 参数
  if [[ "$7" == "--sample_size" ]]; then
    SAMPLE_SIZE_FLAG="--sample_size $8"
  else
    SAMPLE_SIZE_FLAG="--sample_size 10"  # 默认值为10
  fi
fi

# 设置Python可执行路径
PYTHON_CMD=python3

# 使用nohup挂载脚本，并将输出重定向到日志文件
nohup $PYTHON_CMD api_requests.py --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" --config "$CONFIG_FILE" --max_workers "$MAX_WORKERS" $DEBUG_FLAG $SAMPLE_SIZE_FLAG > "$LOG_FILE" 2>&1 &
