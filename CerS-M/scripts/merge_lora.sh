#!/bin/bash

MODEL_NAME="./qwen2_5_with_custom_vit"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path output/checkpoint \
    --model-base $MODEL_NAME  \
    --save-model-path output/merged \
    --safe-serialization