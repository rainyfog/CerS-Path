import argparse
from utils import merge_vit_qwen2_5_weights

def merge_vit(args):
    processor, model = merge_vit_qwen2_5_weights(model_base=args.model_base)

    model.save_pretrained(args.save_model_path, safe_serialization=args.safe_serialization)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--safe-serialization", action='store_true')

    args = parser.parse_args()

    merge_vit(args)