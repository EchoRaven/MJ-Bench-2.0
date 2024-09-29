#!/bin/bash

# Set the CUDA devices
export CUDA_VISIBLE_DEVICES=6,7

# Run the swift sft command with nohup and output logs to a file
swift sft \
    --model_type internvl2-2b \
    --model_revision master \
    --dtype bf16 \
    --sft_type full \
    --model_id_or_path /remote_shome/snl/feilong/xiapeng/haibo/videoRM/Internvl/pretrained/InternVL2-2B \
    --custom_train_dataset_path /remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora_json/swift_train.jsonl \
    --custom_val_dataset_path /remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora_json/swift_test.jsonl \
    --learning_rate 4e-5 \
    --output_dir /remote_shome/snl/feilong/xiapeng/haibo/videoRM/finetune_result/Internvl2_2B_base_model_full \
    --max_length 4096 \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --save_strategy "steps" \
    --save_steps 200 \
    --eval_batch_size 1 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --train_dataset_sample -1 \
    --logging_steps 1 \
    --batch_size 1 \
    --max_grad_norm 0.5 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --lazy_tokenize true \
    --preprocess_num_proc 4 \
    --use_flash_attn true