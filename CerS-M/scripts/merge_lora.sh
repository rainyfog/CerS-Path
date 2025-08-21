#!/bin/bash

MODEL_NAME="./qwen2_5_with_custom_vit"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /mnt/hcufs/VLM-WYZ/Qwen2-VL-Finetune/output/finetune_lora_mlp/checkpoint-3482 \
    --model-base $MODEL_NAME  \
    --save-model-path /mnt/hcufs/VLM-WYZ/Qwen2-VL-Finetune/output/finetune_lora_mlp/merge_test-3482 \
    --safe-serialization