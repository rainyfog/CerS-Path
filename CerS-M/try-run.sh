# #!/bin/bash

# # MODEL_NAME="./qwen2_5_with_custom_vit"
# # MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

# export PYTHONPATH=src:$PYTHONPATH

# # python src/merge_custom_vit.py --model-base ./qwen2_5_model_base --save-model-path ./qwen2_5_with_custom_vit --safe-serialization



sh scripts/finetune_lora_mlp.sh

# # sh scripts/merge_lora.sh

# # python -m src.serve.app-try 

# bash scripts/finetune.sh


###############测试#################
#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="./output/merged_qwen2_5"

# MODEL_NAME="output/merge_test-3482-custom_qwen2_5"
# GLOBAL_BATCH_SIZE=128
# BATCH_PER_DEVICE=2
# NUM_DEVICES=1
# GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# export PYTHONPATH=src:$PYTHONPATH

# DS_SKIP_CUDA_CHECK=1 deepspeed --master_port 15666 src/training/train-try-test.py \
#     --use_liger True \
#     --deepspeed scripts/zero3_offload.json \
#     --model_id $MODEL_NAME \
#     --data_path /mnt/hcufs/VLM-WYZ/Data/henet_data_caption2500k_2025-03-09-changed-checked.json \
#     --image_folder /mnt/hcufs/VLM-WYZ/Data/ \
#     --remove_unused_columns False \
#     --freeze_vision_tower True \
#     --freeze_llm True \
#     --tune_merger False \
#     --bf16 True \
#     --fp16 False \
#     --disable_flash_attn2 False \
#     --output_dir output/fft_0912 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $BATCH_PER_DEVICE \
#     --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
#     --image_min_pixels $((512 * 28 * 28)) \
#     --image_max_pixels $((1280 * 28 * 28)) \
#     --learning_rate 1e-5 \
#     --merger_lr 1e-5 \
#     --vision_lr 2e-6 \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --gradient_checkpointing True \
#     --report_to tensorboard \
#     --lazy_preprocess True \
#     --save_strategy "steps" \
#     --save_steps 2 \
#     --save_total_limit 10 \
#     --dataloader_num_workers 4