import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from roi import Classifier, DATA_REPO, image_transform


class Tiles(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = image_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB')), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'})
    if not paths:
        raise ValueError('No input images found')
    checkpoint = args.checkpoint or hf_hub_download(DATA_REPO, 'classifier.pt', repo_type='dataset')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    names = saved['classes']
    model = Classifier(len(names), args.backbone)
    model.classifier.load_state_dict(saved['state_dict'], strict=True)
    model.eval().to(args.device)
    loader = DataLoader(Tiles(paths), batch_size=args.batch_size, num_workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.writer(f)
        writer.writerow(['image', 'class', 'confidence', 'probabilities'])
        for images, indices in loader:
            probabilities = model(images.to(args.device)).softmax(1).cpu().tolist()
            for index, values in zip(indices.tolist(), probabilities):
                label = max(range(len(names)), key=values.__getitem__)
                writer.writerow([str(paths[index]), names[label], values[label], json.dumps(values)])
    print(f'Saved {len(paths)} predictions to {args.output}')


if __name__ == '__main__':
    main()
