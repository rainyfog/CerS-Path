import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from roi import Classifier, class_names, data_directory, image_transform


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--backbone', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    directory = args.data or data_directory()
    augmented = ImageFolder(directory, transform=image_transform(True))
    plain = ImageFolder(directory, transform=image_transform())
    names = class_names(directory, augmented.classes)
    count = len(augmented)
    train_count, validation_count = int(count * 0.7), int(count * 0.1)
    if min(train_count, validation_count, count - train_count - validation_count) < 1:
        raise ValueError('At least ten images are required')
    indices = torch.randperm(count).tolist()
    train = Subset(augmented, indices[:train_count])
    validation = Subset(augmented, indices[train_count:train_count + validation_count])
    test = Subset(plain, indices[train_count + validation_count:])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=32, num_workers=args.workers)
    model = Classifier(len(names), args.backbone).to(args.device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=20, cooldown=10)
    criterion = torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True)
    checkpoint = args.output / 'classifier.pt'
    selected_accuracy = -1.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(args.device)), labels.to(args.device))
            loss.backward()
            optimizer.step()
        accuracy = evaluate(model, validation_loader, args.device)
        scheduler.step(100 * accuracy)
        if accuracy > selected_accuracy:
            selected_accuracy = accuracy
            state = {name: value.detach().cpu().clone() for name, value in model.classifier.state_dict().items()}
            torch.save({'state_dict': state, 'classes': names}, checkpoint)
        print(f'Epoch {epoch + 1}/{args.epochs}: validation accuracy {accuracy:.4f}', flush=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.classifier.load_state_dict(saved['state_dict'], strict=True)
    print(f'Test accuracy: {evaluate(model, test_loader, args.device):.4f}')
    print(f'Saved {checkpoint}')


if __name__ == '__main__':
    main()
