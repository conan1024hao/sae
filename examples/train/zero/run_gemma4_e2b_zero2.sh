#!/bin/bash

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export WANDB_PROJECT="multimodal-sae"
export RUN_NAME="gemma-4-e2b-vision-projection-sae-finevisionmax-500k"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"

# Gemma 4's counterpart of Qwen2.5-VL's `model.visual.merger.mlp.2` ("projection-mlp2"):
# the bias-free Linear(768 -> 1536) inside `Gemma4MultimodalEmbedder` that projects the
# pooled vision soft tokens into language-model embedding space.
TARGET_MODULE="model.embed_vision.embedding_projection"

# Notes on two settings that differ from run_qwen25_vl_3b_zero2.sh:
#
# --per_device_train_batch_size 1 / --gradient_accumulation_steps 2
#   Gemma 4 E2B is 5.1B params against Qwen2.5-VL-3B's 3.9B, and a 30-step 8xRTX5090
#   run at micro-batch 2 peaked at 28.9 of 32.6 GiB -- too close to OOM to leave running
#   for 31250 steps. Halving the micro-batch and accumulating twice keeps the effective
#   batch (8 x 1 x 2 = 16) and the total sample count (16 x 31250 = 500k) identical to
#   the Qwen run.
#
# --dataloader_num_workers 0
#   Must stay 0. Stream-position checkpointing (sae.trainer.dataset_state) reads the
#   dataset object in this process; a DataLoader worker would consume rows in its own
#   copy, so the recorded position would never advance. It also removes the worker-side
#   HTTP client whose failure killed the first attempt at step 6203.

torchrun --nproc_per_node="8" --nnodes="1" --node_rank="0" --master_addr="127.0.0.1" --master_port="1234" \
    src/sae/launch/train.py \
    --dataset_path HuggingFaceM4/FineVisionMax \
    --split train \
    --image_key images \
    --text_key texts \
    --report_to wandb \
    --model-path "google/gemma-4-E2B-it" \
    --bf16 \
    --target_modules $TARGET_MODULE \
    --dataloader_num_workers 0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --accelerator_config '{"dispatch_batches":false}' \
    "$@"
