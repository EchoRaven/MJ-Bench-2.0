#!/bin/bash

# 默认参数的shell脚本 - 使用nohup在后台训练
nohup accelerate launch train.py \
  --path "OpenGVLab/InternVL2-2B" \
  --train_save_dir "dataset/saved_train_dataset" \
  --test_save_dir "dataset/saved_test_dataset" \
  --output_dir "./results" \
  --logging_dir "./logs" \
  --epochs 3 \
  --batch_size 8 \
  --lr 5e-5 \
  --warmup_steps 500 \
  --logging_steps 10 \
  --evaluation_strategy "epoch" > train.log 2>&1 &
