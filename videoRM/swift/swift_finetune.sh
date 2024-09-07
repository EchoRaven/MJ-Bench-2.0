#!/bin/bash

# Set the CUDA devices
export CUDA_VISIBLE_DEVICES=1,5

# Run the swift sft command with the provided arguments
swift sft \
    --model_type internvl2-2b \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --dtype AUTO \
    --model_id_or_path /remote_shome/snl/feilong/xiapeng/haibo/videoRM/Internvl/pretrained/InternVL2-2B \
    --custom_train_dataset_path /remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora_json/swift_train.jsonl \
    --custom_val_dataset_path /remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora_json/swift_test.jsonl \
    --learning_rate 4e-5 \
    --output_dir /remote_shome/snl/feilong/xiapeng/haibo/videoRM/finetune_result/Internvl2_2B_base_model \
    --max_length 4096 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --deepspeed "zero_stage3_config.json" \
    --num_train_epochs 1 \
    --save_steps 200 \
    --train_dataset_sample -1 \
    --logging_steps 1 \
    --batch_size 1 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --max_grad_norm 0.5 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"
