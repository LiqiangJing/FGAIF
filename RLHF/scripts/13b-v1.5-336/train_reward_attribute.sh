#!/bin/bash

set -e
set -x

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR="/home/lxj220018/llava-rlhf/data"
export MODEL_DIR="/home/lxj220018/llava-rlhf/checkpoints/baseline"
export MODEL_SAVEW_DIR="/home/lxj220018/llava-rlhf/checkpoints"
export PYTHONPATH="$PWD:$PYTHONPATH"
#export GPUS_PER_NODE=8
export GPUS_PER_NODE=4
#export OMP_NUM_THREADS=8
export OMP_NUM_THREADS=4


# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14-336
LM_MODEL_NAME=LLaVA-RLHF-13b-v1.5-336/sft_model

# DATA CONFIG
#PREFERENCE_DATA=llava_7b_v1_preference.json
#PREFERENCE_DATA=fg_reward_data.json
#PREFERENCE_DATA=data/fg_AI_reward_data.json
PREFERENCE_DATA=data/fg_atomic_reward_data.json

# SAVE CONFIG
#MODEL_NAME=LLaVA-ATT-AI-RM-13b-v1.5-336-lora-padding
MODEL_NAME=LLaVA-ATT-Atomic-RM-13b-v1.5-336-lora-padding

# TRAINING CONFIG
NUM_EPOCHS=100
LEARNING_RATE=2e-5
#BATCH_SIZE=4
#GRAD_ACCUMULATION=1
BATCH_SIZE=4
GRAD_ACCUMULATION=2

torchrun \
    --standalone \
    --nnodes=1 \
    --master_port=25641 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_fine_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
    --image_folder $DATA_DIR/coco/train2014 \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --eval_dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 400 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_SAVEW_DIR/$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 15 \
    --save_strategy "steps" \
    --save_steps 15 \
    --save_total_limit 10 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --image_to_caption_file "$DATA_DIR/train2014_image_to_caption.json" \
    --reward_prompt_file "./prompts/fact_rlhf_reward_prompt.txt" \
    --image_aspect_ratio 'pad' \
    --reward_type 1
