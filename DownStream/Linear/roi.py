import json
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download
import timm
import torch
from torchvision import transforms

DATA_REPO = 'Rainy-fog/Cervical-Histopathology-Tile-Classification'


def image_transform(training=False):
    steps = [transforms.Resize(224)]
    if training:
        steps += [
            transforms.CenterCrop(224),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(30),
                transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            ]),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.2, hue=0.5),
                transforms.RandomAffine(degrees=0, shear=15),
            ], p=0.2),
        ]
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def data_directory():
    target = Path('data') / 'cervical_tiles'
    if not (target / 'images').is_dir():
        archive = hf_hub_download(DATA_REPO, 'tiles.zip', repo_type='dataset')
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                destination = (target / member.filename).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise ValueError('Invalid archive path')
            z.extractall(target)
    return target / 'images'


def class_names(directory, fallback):
    path = Path(directory).parent / 'classes.json'
    names = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else fallback
    if len(names) != len(fallback) or not all(isinstance(x, str) for x in names):
        raise ValueError('classes.json must contain one name per class directory')
    return names


class Classifier(torch.nn.Module):
    def __init__(self, classes, backbone=None):
        super().__init__()
        weights = backbone or hf_hub_download('Rainy-fog/CerS-Path', 'weights/CerS-Path.pth')
        self.encoder = timm.create_model('vit_large_patch16_224', pretrained=False,
            img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        self.encoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.encoder.requires_grad_(False)
        self.classifier = torch.nn.Linear(1024, classes)

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)
        return self.classifier(features)
