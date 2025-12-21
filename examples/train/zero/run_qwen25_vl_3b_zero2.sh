#!/bin/bash

export WANDB_API_KEY="5b8322df11b04a8895325a5bf6ef8cbba0dd64a2"
export WANDB_PROJECT="multimodal-sae"
export RUN_NAME="qwen2.5-vl-3b-vision-block30-sae-finevisionmax-500k"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

TARGET_MODULE="model.visual.blocks.30.mlp.down_proj"


torchrun --nproc_per_node="8" --nnodes="1" --node_rank="0" --master_addr="127.0.0.1" --master_port="1234" \
    src/sae/launch/train.py \
    --dataset_path HuggingFaceM4/FineVisionMax \
    --split train \
    --image_key images \
    --text_key texts \
    --report_to wandb \
    --model-path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --bf16 \
    --target_modules $TARGET_MODULE \
    --dataloader_num_workers 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --max_steps 31250 \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --output_dir checkpoints/$RUN_NAME \
    --save_total_limit 5 \
    --deepspeed ./examples/train/zero/zero2.json \
    --num_latents 32768 \
    --k 64 \
    --dead_tokens_threshold 100000 \
    --run_name $RUN_NAME \
    --streaming \
    --accelerator_config '{"dispatch_batches":false}'