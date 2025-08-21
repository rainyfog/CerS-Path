import torch
from torch.optim import Adam
import time
import os
import albumentations as albu
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.base import modules as base_modules
from segmentation_models_pytorch.utils import functional as F
import segmentation_models_pytorch as smp

# Data Augmentation and Preprocessing
# train_augmentation = albu.Compose([
#     # albu.RandomCrop(height=512, width=256),
#     albu.RandomCrop(height=512, width=512),
#     albu.HorizontalFlip(),
#     albu.VerticalFlip(),
#     albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# val_augmentation = albu.Compose([
#     albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

### 图像增强
IMAGE_SIZE=512
def get_training_augmentation():
    train_transform = [
        albu.Resize ( IMAGE_SIZE, IMAGE_SIZE ),
        albu.HorizontalFlip ( p=0.5 ),
        albu.ShiftScaleRotate ( scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0 ),
        albu.GaussNoise ( p=0.2 ),
        albu.Perspective ( p=0.5 ),

        albu.OneOf (
            [
                albu.CLAHE ( p=1 ),
                albu.RandomBrightnessContrast ( p=1 ),
                albu.RandomGamma ( p=1 ),
            ],
            p=0.9,
        ),
        albu.OneOf (
            [
                albu.Sharpen ( p=1 ),
                albu.Blur ( blur_limit=3, p=1 ),
                albu.MotionBlur ( blur_limit=3, p=1 ),
            ],
            p=0.9,
        ),
        albu.OneOf (
            [
                # albu.RandomContrast ( p=1 ),
                # albu.rand
                albu.HueSaturationValue ( p=1 ),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose ( train_transform )

def get_validation_augmentation():
    """调整图像使得图片的分辨率长宽能被32整除"""
    test_transform = [
        albu.Resize ( IMAGE_SIZE, IMAGE_SIZE )
    ]
    return albu.Compose ( test_transform )
import cv2
import os
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset

def to_tensor(x, **kwargs):
    return x.transpose ( 2, 0, 1 ).astype ( 'float32' )
    
def get_preprocessing():
    """进行图像预处理操作

    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Lambda ( image=to_tensor, mask=to_tensor ),
    ]
    return albu.Compose ( _transform )

class Dataset(Dataset):
    CLASSES = ['tissue']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=get_preprocessing()):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace(".jpg", ".png")) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes else [0]  # Default to tissue class
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask[mask == 255] = 1  # Binary mask

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


## 计算 IoU
class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation="sigmoid", ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = base_modules.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
            
def calculate_iou(pred, target, threshold=0.5):
    cal = IoU()
    iou_score = cal(pred,target)
    return iou_score # 返回每个 batch 的 IoU 平均值

# 计算 Dice 系数
def calculate_dice(pred, target, activation_fn=torch.sigmoid, threshold=0.5):
    """
    计算 Dice 系数，可以选择在输入时应用激活函数。
    
    参数:
    - pred: 模型的预测输出 (未经过激活的原始值)
    - target: 真实标签
    - activation_fn: 激活函数，默认为 None（不使用激活函数）
    - threshold: 用于二值化预测的阈值，默认为 0.5
    
    返回:
    - Dice 系数的平均值
    """
    # 如果传入了激活函数，则对预测进行激活
    if activation_fn:
        pred = activation_fn(pred)
    
    # 二值化预测值
    pred = (pred > threshold).float()
    target = target.float()

    # 计算交集部分
    intersection = (pred * target).sum(dim=(1, 2, 3))
    
    # 计算 Dice 系数
    dice = (2. * intersection + 1e-6) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6)
    
    # 返回每个 batch 的 Dice 系数平均值
    return dice.mean().item()




# 训练函数，添加 IoU 计算
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0  # 初始化 IoU
    # Wrap train_loader with tqdm to show progress bar
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    
    for images, masks in progress_bar:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算训练损失
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算 IoU
        iou = calculate_iou(outputs, masks)
        running_iou += iou

        # 更新进度条显示
        progress_bar.set_postfix(loss=loss.item(), IoU=iou)

    avg_loss = running_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    return avg_loss, avg_iou

# 验证函数，添加 Dice 和 IoU 计算
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0  # 初始化 Dice
    running_iou = 0.0   # 初始化 IoU
    with torch.no_grad():
        # Wrap val_loader with tqdm to show progress bar
        progress_bar = tqdm(val_loader, desc="Validation", unit="batch")
        
        for images, masks in progress_bar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            
            # 计算验证损失
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # 计算 Dice 和 IoU
            dice = calculate_dice(outputs, masks)
            iou = calculate_iou(outputs, masks)
            running_dice += dice
            running_iou += iou

            # 更新进度条显示
            progress_bar.set_postfix(loss=loss.item(), Dice=dice, IoU=iou)

    avg_loss = running_loss / len(val_loader)
    avg_dice = running_dice / len(val_loader)
    avg_iou = running_iou / len(val_loader)
    return avg_loss, avg_dice, avg_iou



# 主循环
def main():
    # Dataset paths
    train_images_dir = "/home/rainyfog/code/Tissue-seg/zz_Code-tissue-seg/DATA/tissue_1024_cropped_images/train/images"
    train_masks_dir = "/home/rainyfog/code/Tissue-seg/zz_Code-tissue-seg/DATA/tissue_1024_cropped_images/train/masks"
    val_images_dir = "/home/rainyfog/code/Tissue-seg/zz_Code-tissue-seg/DATA/tissue_1024_cropped_images/valid/images"
    val_masks_dir = "/home/rainyfog/code/Tissue-seg/zz_Code-tissue-seg/DATA/tissue_1024_cropped_images/valid/masks"

    # Dataset and DataLoader
    from torch.utils.data import Subset
    train_dataset = Dataset(train_images_dir, train_masks_dir, classes=['tissue'], augmentation=get_training_augmentation())

    val_dataset = Dataset(val_images_dir, val_masks_dir, classes=['tissue'], augmentation=get_validation_augmentation())


    # 获取训练集和验证集的大小
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    # 计算取10分之一的大小
    train_subset_size = train_size // 10
    val_subset_size = val_size // 10

    # 随机选择训练集和验证集中的 10% 的索引
    train_indices = torch.randperm(train_size).tolist()[:train_subset_size]
    val_indices = torch.randperm(val_size).tolist()[:val_subset_size]

    # 创建训练集和验证集的子集
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,num_workers=8)

    # Initialize model
    from PFM_Seg_Models import PFM_Seg_Model
    model = PFM_Seg_Model(PFM_name='UNI', PFM_weights_path='/home/rainyfog/code/Extra_features/ckpt/UNI/pytorch_model.bin', emb_dim=1024, frozen_PFM=True,num_classes=1)
    
    # 打印出需要更新的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

    # Loss function and optimizer
    
    criterion = smp.losses.DiceLoss(mode='binary',from_logits=True)
    
    optimizer = optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    num_epochs = 20
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_iou = train(model, train_loader, criterion, optimizer, device)
        
        # Validate the model
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, "
              f"Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()