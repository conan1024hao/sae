#!/bin/bash

export WANDB_API_KEY="5b8322df11b04a8895325a5bf6ef8cbba0dd64a2"
export WANDB_PROJECT="multimodal-sae"
export RUN_NAME="test-run"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

TARGET_MODULE="model.language_model.layers.27.mlp.down_proj"


torchrun --nproc_per_node="8" --nnodes="1" --node_rank="0" --master_addr="127.0.0.1" --master_port="1234" \
    src/sae/launch/train.py \
    --dataset_path lmms-lab/LLaVA-OneVision-Data-SAE \
    --split train \
    --subset "CLEVR-Math(MathV360K)" \
    --image_key images \
    --text_key text \
    --report_to wandb \
    --model-path "Qwen/Qwen3-VL-4B-Instruct" \
    --bf16 \
    --target_modules $TARGET_MODULE \
    --dataloader_num_workers 1 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 1000 \
    --output_dir checkpoints/$RUN_NAME \
    --save_total_limit 5 \
    --deepspeed ./examples/train/zero/zero2.json \
    --num_latents 65536 \
    --k 128 \
    --dead_tokens_threshold 100000 \
    --run_name $RUN_NAME \
