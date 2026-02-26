#!/bin/bash

DATASET_DIR="../dataset/HAM10000/train"
MODEL_ID="PixArt-alphaPixArt-XL-2-512x512"
FEATURE_BASE_DIR="../dataset/HAM10000/extracted_text_features"
OUTPUT_DIR="../pixart_alphaPixArt_XL_LoRA/text_projection1"

export CUDA_VISIBLE_DEVICES=3

accelerate launch --mixed_precision="bf16" --num_processes=1 --main_process_port=29500 train_medart_lora_text_projection1.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --train_data_dir=$DATASET_DIR \
  --feature_base_dir=$FEATURE_BASE_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=15 \
  --rank=8 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --max_grad_norm=1.0 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --checkpoints_total_limit=10 \
  --M_times=20 \
  --N_steps=500




