#!/bin/bash
#SBATCH --job-name=q71h728_123
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-cpu=9200
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err


python eval_reward_model.py \
    --rlhf_type rm \
    --model_type  internvl2-2b \
    --model_id_or_path ../Internvl/pretrain/InternVL2-2B \
    --sft_type  lora \
    --output_dir ../finetune_result/alignment_expert_Internvl2_2B_base_model_lora_single_rm \
    --custom_train_dataset_path ../dataset/swift_json_file_single_rm/swift_alignment_train.jsonl \
    --custom_val_dataset_path ../dataset/swift_json_file_single_rm/swift_alignment_test.jsonl \
    --resume_from_checkpoint ../finetune_result/alignment_expert_Internvl2_2B_base_model_lora_single_rm/internvl2-2b/v8-20241103-024332/checkpoint-1232 \
    --num_train_epochs  2  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  4  \
    --eval_steps 10000000 \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  8  \
    --warmup_ratio  0.03  \
    --save_total_limit  2 \
    --lazy_tokenize true \
    --preprocess_num_proc 4 \
    --use_flash_attn true