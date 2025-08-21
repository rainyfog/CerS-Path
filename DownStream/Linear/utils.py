# -*-coding:utf-8-*-
import os
import sys
import re
import datetime
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import random
from torch.utils.data import Dataset
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import pandas as pd
import timm
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset
import random
import time
def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_class=args.num_classes)
    elif args.net == 'mynet':
        from models.mynet import create_mynet
        net = create_mynet(num_class=args.num_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_class=args.num_classes)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_class=args.num_classes)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_class=args.num_classes)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(num_class=args.num_classes)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(class_names=3)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None,seed=None,label_ratio=1.5):
        """
        初始化自定义数据集
        :param image_dir: 图片文件夹路径
        :param transform: 图像变换（如需数据增强）
        """
        self.image_dir = image_dir
        self.transform = transform
        self.seed = seed
        self.label_ratio = label_ratio
        self.image_files = self.filter_files_by_size(80)
        # self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.image_files = self.balance_samples()
        print("完成平衡采样，比例是“{label_ratio}")

    def filter_files_by_size(self, min_size):
        """
        过滤出尺寸大于指定大小的文件
        :param min_size: 最小尺寸
        :return: 过滤后的文件列表
        """
        files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        filtered_files = []
        for f in files:
            try:
                # 从文件名中提取尺寸信息
                size_str = f.split("Size-")[-1].split("_")[0]
                size = int(size_str)
                # 判断尺寸是否大于指定大小
                if size > min_size:
                    filtered_files.append(f)
            except ValueError:
                # 如果解析尺寸失败，忽略该文件
                print(f"无法解析文件尺寸：{f}")
        return filtered_files
        
    def balance_samples(self):
        # 设置随机种子
        if self.seed is not None:
            random.seed(self.seed)
        
        # 将带有 "Label-0" 和 "Label-1" 的文件分别存入列表
        label_0_files = [f for f in self.image_files if "Label-0" in f]
        label_1_files = [f for f in self.image_files if "Label-1" in f]
        
        # 打印原始的标签分布
        print(f"原始样本数量：Label-0 = {len(label_0_files)}, Label-1 = {len(label_1_files)}")
        
        # 计算抽样数量
        if len(label_0_files) > len(label_1_files):
            target_size = int(len(label_1_files) * self.label_ratio)
            label_0_files = random.sample(label_0_files, min(target_size, len(label_0_files)))
        else:
            target_size = int(len(label_0_files) * self.label_ratio)
            label_1_files = random.sample(label_1_files, min(target_size, len(label_1_files)))
        
        # 打印抽样后的标签分布
        print(f"抽样后样本数量：Label-0 = {len(label_0_files)}, Label-1 = {len(label_1_files)}")

        # 合并平衡后的样本列表
        balanced_files = label_0_files + label_1_files
        random.shuffle(balanced_files)  # 打乱顺序
        return balanced_files
    
    def __len__(self):
        # 返回数据集大小
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        :param idx: 索引
        :return: 图像张量和对应的标签
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 读取图像
        image = Image.open(img_path).convert("RGB")
        
        # 提取标签
        label_str = img_name.split("Label-")[-1].split(".")[0]  # 取文件名中 'Label-' 后的部分
        label = int(label_str)  # 转换为整数标签
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


    # 定义你的模型
    
def get_training_dataloader(path, mean, std, input_size=256, batch_size=16, prop=0.1,num_worker=16,dying_rate = 10,seeds=2024,transform_name = "train"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if transform_name=="train":
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomChoice([transforms.RandomHorizontalFlip(0.5),
                                                        transforms.RandomRotation(30),
                                                        torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))]),
            # transforms.RandomHorizontalFlip(0.3),
            # # transforms.RandomVerticalFlip(0.3),
            # transforms.RandomRotation(30),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.3, contrast=0.2, hue=0.5),
                                                        transforms.RandomAffine(degrees=0, shear=15)]), p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    print(transform_train)
    # # trainset = CustomImageDataset(path, transform=transform_train,seed=seeds)
    # trainset = torchvision.datasets.ImageFolder(path,transform=transform_train)
    
    # all_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=num_worker)
    
    # length = len(trainset)
    
    # train_size, validate_size,test_size = int(length - 3*int(prop * length)), 1*int(prop * length), 2*int(prop * length) 
    # train_db, val_db,test_db= torch.utils.data.random_split(trainset, [train_size, validate_size,test_size])
    
    # 设置固定的随机种子
    torch.manual_seed(seeds)

    trainset = torchvision.datasets.ImageFolder(path, transform=transform_train)

    all_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    length = len(trainset)

    # 确定划分比例
    train_size = length - 3 * int(prop * length)
    validate_size = int(prop * length)
    test_size = 2 *int( prop * length)

    # 使用固定的随机种子来确保每次划分相同
    train_db, val_db, test_db = torch.utils.data.random_split(trainset, [train_size, validate_size, test_size], generator=torch.Generator().manual_seed(seeds))
    
    from torch.utils.data import Subset
    train_subset_size = len(train_db) // dying_rate
    train_subset_indices = torch.randperm(len(train_db))[:train_subset_size]
    # 创建 Subset 数据集
    train_subset_dataset = Subset(train_db, train_subset_indices)

    valid_subset_size = len(val_db) // dying_rate
    valid_subset_indices = torch.randperm(len(val_db))[:valid_subset_size]
    valid_subset_dataset = Subset(val_db, valid_subset_indices)
    
    
    test_subset_size = len(test_db) // dying_rate
    test_subset_indices = torch.randperm(len(test_db))[:test_subset_size]
    test_subset_dataset = Subset(test_db, test_subset_indices)
    
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    # shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(
        train_subset_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_worker)
    # 验证集
    val_loader = torch.utils.data.DataLoader(
        valid_subset_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_worker)
    # 验证集
    test_loader = torch.utils.data.DataLoader(
        test_subset_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_worker)

    return train_loader, val_loader, test_loader, all_loader


def get_training_dataloader_noaug(path, mean, std, input_size=256, batch_size=16, prop=0.8):
    """ return training dataloader with no augments
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = torchvision.datasets.ImageFolder(path, transform=transform_train)
    length = len(trainset)
    train_size, validate_size = int(prop * length), int(length - int(prop * length))
    train_db, val_db = torch.utils.data.random_split(trainset, [train_size, validate_size])
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size, shuffle=True, num_workers=16)
    # 验证集
    val_loader = torch.utils.data.DataLoader(
        val_db,
        batch_size=batch_size, shuffle=True, num_workers=16)
    return train_loader, val_loader


def get_test_dataloader(path,mean, std, input_size=224, batch_size=16, num_workers=16, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_set = torchvision.datasets.ImageFolder(path,
                                                transform=transform_test)
    test_loader = DataLoader(
        test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)

    return test_loader