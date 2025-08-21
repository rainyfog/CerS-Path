import argparse
from utils import get_model_name_from_path, load_pretrained_model

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_path=args.model_path, model_base=args.model_base,
                                             model_name=model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path, safe_serialization=args.safe_serialization)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/hcufs/VLM-WYZ/Qwen2-VL-Finetune/output/finetune_lora_mlp/checkpoint-3482")
    parser.add_argument("--model-base", type=str, default="./qwen2_5_with_custom_vit")
    parser.add_argument("--save-model-path", type=str, default="/mnt/hcufs/VLM-WYZ/Qwen2-VL-Finetune/output/try")
    parser.add_argument("--safe-serialization", action='store_true')

    args = parser.parse_args()

    merge_lora(args)