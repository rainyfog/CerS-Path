# Fine-tuning Qwen2.5-VL 7B with custom Vit

## Installation

Install the required packages using `environment.yaml`.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate qwen2
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Model Preparation
In this step, you need to merge custom pretrained Vision Transformer (ViT) with Qwen2.5-VL 7B model. You can use the `src/merge_custom_vit.py` script to merge the models.
### Step 1: 
Change the config in `./qwen2_5_model_base/config.json`, set the vit state dict path in `vision_config.pretrained_model_path`.
```json
{
  ...
  "vision_config": {
    ...
    "pretrained_model_path": "/path/to/vit_model"
  }
}
```  
###Step 2:
merge the vit with qwen2.5-vl 7B model and save new state dict.
```bash
python src/merge_custom_vit.py --model_base ./qwen2_5_model_base --save-model-path ./qwen2_5_with_custom_vit --safe-serialization
```


## Training

### Step 1: fine-tuning lora mlp and vision tower
set the `--data_path` and `--image_folder` in `scripts/finetune_lora.sh` to your own json data path and image folder path. Then run the following command to fine-tune the mlp and vit with LoRA.
```bash
sh scripts/finetune_lora_mlp.sh
```
if you don't want to fine-tune the vision tower, you can set `freeze_vision_tower True` in `finetune_lora_mlp.sh`.


### Step 2: merge the lora weights with qwen2.5-vl 7B
set the `--model-path` and `--save-model-path` in `scripts/merge_lora.sh` to the path of the model saved in step 1 and the target output path. **make sure the output path contain keyword: "qwen2_5"**. Then run the following command to merge the lora weights with qwen2.5-vl 7B.
#### 
```bash
sh scripts/merge_lora.sh
```

### Step 3: full fine-tuning the merged model
set the `MODEL_NAME` to the `save-model-path` in step 2. Then set the `--data_path` and `--image_folder` in `scripts/finetune.sh` to your own json data path and image folder path. Then run the following command to fine-tune the merged model.

```bash
bash scripts/finetune.sh
```

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Qwen2-VL model. **(Required)**
- `--use_liger` (bool): Option for using liger kernel to save memory.
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_merger` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_min_pixels` (int): Option for minimum input tokens for image.
- `--image_max_pixles` (int): Option for maximum maxmimum tokens for image.
- `--video_min_pixels` (int): Option for minimum input tokens for video.
- `--video_max_pixles` (int): Option for maximum maxmimum tokens for video.
- `--lora_enable` (bool): Option for using LoRA.
- `--vision_lora` (bool): Option for including `vision_tower` in LoRA module. `lora_enable` should be `True` to use this option.
- `--use_dora` (bool): Option for using DoRA instead of LoRA. `lora_enable` should be `True` to use this option.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): Awesome pretrained MLLM based on Qwen2.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Collection of Tirton kernels designed specifically for LLM training.
