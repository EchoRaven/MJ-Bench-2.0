#!/bin/bash
#SBATCH --job-name=moe
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-cpu=9200
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err

swift sft \
    --model_type internvl2-4b \
    --model_revision master \
    --sft_type lora \
    --dtype bf16 \
    --model_id_or_path ../Internvl/pretrain/InternVL2-4B \
    --custom_train_dataset_path ../dataset/swift_json_file/swift_cc_train.jsonl \
    --custom_val_dataset_path ../dataset/swift_json_file/swift_cc_test.jsonl \
    --learning_rate 4e-5 \
    --output_dir ../finetune_result/cc_expert_Internvl2_4B_base_model_lora \
    --max_length 4096 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --eval_batch_size 1 \
    --learning_rate 4e-5 \
    --num_train_epochs 2 \
    --save_steps 200 \
    --lora_rank 6 \
    --train_dataset_sample -1 \
    --logging_steps 1 \
    --batch_size 1 \
    --lora_rank 6 \
    --max_grad_norm 0.5 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --lazy_tokenize true \
    --preprocess_num_proc 4 \
    --use_flash_attn true
