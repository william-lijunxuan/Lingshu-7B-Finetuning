#!/bin/bash

# Distributed training configuration
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=/home/william/model/Lingshu-7B-Finetuning/qwenvl/scripts/zero3.json

# Model configuration
llm=/home/william/model/Qwen3-VL-8B-Thinking  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=/home/william/model/Lingshu-7B-Finetuning/qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=derm1m

# Output configuration
run_name="qwen3-vl-8b-baseline"
output_dir=./outputqwen3vl

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --lora_enable True\
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}