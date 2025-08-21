#coding=utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import timm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,balanced_accuracy_score
from PIL import ImageFile
from numpy import trapz
from utils import *
# from utils import get_training_dataloader
import argparse
import time
from torch.optim import lr_scheduler
from torchvision import models
import gc
import copy
import ssl 
from tqdm import tqdm
ssl._create_default_https_context = ssl._create_unverified_context

# from zz_mynet import Mynet
# from metric_visiualize import draw_accuracy_curve, draw_loss_curve
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义你的模型
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # 定义模型结构
        if args.model_name =="UNI":
            local_dir = "UNI"
            # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
            # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
            model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            weights = torch.load(args.model_path,map_location="cpu")
            # weights = weights["backbone"]
            model.load_state_dict(weights,  strict=True)
            self.encoder = model
            self.decoder = nn.Linear(in_features=1024,out_features=args.num_classes)
            
        elif args.model_name =="ST":
            # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
            # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
            model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            model.load_state_dict(torch.load("/data_sdd/wyz/Weight/ST/eval/weights/PathOrchestra_V1.0.0.bin", map_location="cpu"), strict=True)
            self.encoder = model
            self.decoder = nn.Linear(in_features=1024,out_features=args.num_classes)
            
        if args.model_name =="Cervix":
            # local_dir = "ckpt"
            # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
            # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
            model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            weights = torch.load(args.model_path,map_location="cpu")
            # weights = weights["backbone"]
            model.load_state_dict(weights,  strict=True)
            self.encoder = model
            self.decoder = nn.Linear(in_features=1024,out_features=args.num_classes)
            
        if args.model_name =="CONCHV15":
            from TITAN.configuration_titan import ConchConfig
            from TITAN.conch_v1_5 import build_conch
            conch_config  = ConchConfig()
            model,conch_1_5_eval_transform = build_conch(conch_config)
            self.encoder = model
            self.decoder = nn.Linear(in_features=768,out_features=args.num_classes)
            
        if args.model_name =="virchow2":
            from timm.layers import SwiGLUPacked
            virchow_v2_config = {
            "img_size": 224,
            "init_values": 1e-5,
            "num_classes": 0,
            "mlp_ratio": 5.3375,
            "reg_tokens": 4,
            "global_pool": "",
            "dynamic_img_size": True}
            model = timm.create_model("vit_huge_patch14_224", pretrained=False,mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,**virchow_v2_config)
            state_dict = torch.load("/home/wyz/code/Extra_features/ckpt/virchow2/Virchow_2_weights/pytorch_model.bin", map_location="cpu",weights_only=True)
            model.load_state_dict(state_dict, strict=True)        
            self.encoder = model
            self.decoder = nn.Linear(in_features=1280,out_features=args.num_classes)
            
    
    def forward(self, x):
        x= self.encoder(x)
        if args.model_name=="virchow2":
            x = x[:, 0]    # size: 1 x 1280
            patch_tokens = x[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        x = self.decoder(x)
        return x


def train_and_valid(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    
    num_epochs, lr = args.num_epochs, args.lr
    train_loader, valid_loader,test_loader,all_loader = get_training_dataloader(args.train_dir, mean, std, input_size=args.input_size,
                                                  batch_size=args.batch_size,num_worker=args.num_workers,dying_rate = args.dying_rate,seeds = args.seed)
    model = YourModel().to(device)
    # model = nn.DataParallel(model, device_ids=[ 1, 2, 3])
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=False,
                                               threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=0, eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    tr_loss, va_loss, tr_acc_all, va_acc_all,va_f1_all,va_recall_all, val_precision,epoch_message = [], [], [], [], [],[],[],[]
    
    best_loss = 0.1
    best_va = 0
    best_model_wts = []
    valid = '\n'
    save_path = f"{args.save_root}_{args.net}_{args.seed}"
    os.makedirs(save_path,exist_ok=True)
    with open ( os.path.join(save_path,"train_messages"+args.net+".txt"), "a" ) as f:
        f.write ( valid )
        f.write ( valid )
        f.write ( 'start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start' )
        f.write ( valid )

    for epoch in range(num_epochs):
        epoch_message.append(epoch)
        running_loss, epoch_va_loss = 0.0, 0.0
        train_pred, train_true, validate_pred, validate_true,test_pred, test_true = [], [], [], [],[],[]
        print('-' * 10, 'Epoch {}/{}'.format(epoch + 1, num_epochs), '-' * 10)
        time_start = time.time()

        # training part
        model.train()
        from tqdm import tqdm
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # outputs = F.softmax(outputs, dim=1)
            batch_train_loss = criterion(outputs, labels)
            batch_train_loss.backward()
            optimizer.step()
            running_loss += batch_train_loss.item()

            outputs = F.softmax(outputs, dim=1)
            train_true.extend(labels)
            train_pred.extend(torch.argmax(outputs, dim=1))
            len_data = i+1
        # print(len_data)=688
        running_loss = running_loss/ len_data
        acc_tr = 100*accuracy_score(torch.tensor(train_true).cpu(), torch.tensor(train_pred).cpu())
        tr_acc_all.append(acc_tr)
        tr_loss.append(running_loss)

        # validation part
        model.eval()

        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_loader, 0), total=len(valid_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # outputs = F.softmax(outputs, dim=1)
                batch_validate_loss = criterion(outputs, labels)
                epoch_va_loss += batch_validate_loss.item()
                outputs = F.softmax(outputs, dim=1)
                validate_true.extend(labels)
                validate_pred.extend(torch.argmax(outputs, dim=1))
                len_data = i+1
            epoch_va_loss = epoch_va_loss / len_data
            # scheduler.step(epoch_va_loss)
            acc_va = 100*accuracy_score(torch.tensor(validate_true).cpu(), torch.tensor(validate_pred).cpu())
            va_acc_all.append(acc_va)
            f1 = f1_score(torch.tensor(validate_true).cpu(), torch.tensor(validate_pred).cpu(), average='macro')
            va_f1_all.append(f1)
            recall = recall_score(torch.tensor(validate_true).cpu(), torch.tensor(validate_pred).cpu(), average='macro')
            va_recall_all.append(recall)
            precision = precision_score(torch.tensor(validate_true).cpu(), torch.tensor(validate_pred).cpu(), average='macro')
            val_precision.append(precision)
            scheduler.step(acc_va)
            va_loss.append(epoch_va_loss)
            
        time_cost = time.time() - time_start
        if acc_va > best_va:
            best_va = acc_va
            best_model_wts = copy.deepcopy(model)
            print("storing best model")
            # best_loss = epoch_va_loss
            best_epoch = epoch
            torch.save(model, os.path.join(save_path,'{}_best.pth'.format(args.net)))
            
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        with open ( os.path.join(save_path,"train_messages"+args.net+".txt"), "a" ) as f:
            f.write ( "第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']) )
            f.write ( valid )
            print('{:.0f}min {:.0f}s,Epoch {}/{}'.format(
            time_cost // 60, time_cost % 60, epoch + 1, num_epochs))
            f.write ( '{:.0f}min {:.0f}s,Epoch {}/{}'.format(time_cost // 60, time_cost % 60, epoch + 1, num_epochs) )
            f.write ( valid )
            print('train_loss = {:.4}, train_accuracy = {:.3f}%'.format(running_loss, acc_tr))
            print('valid_loss = {:.4}, valid_accuracy = {:.3f}%, valid_f1 = {:.3f}, valid_recall={:.3f}, valid_precision ={:.3f} '.format(
                epoch_va_loss, acc_va,f1,recall,precision))
            f.write ( 'train_loss = {:.4}, train_accuracy = {:.3f}%'.format(running_loss, acc_tr))
            f.write ( valid )
            f.write('valid_loss = {:.4}, valid_accuracy = {:.3f}%, valid_f1 = {:.3f}, valid_recall={:.3f}, valid_precision ={:.3f} '.format(
                epoch_va_loss, acc_va,f1,recall,precision))
            f.write ( valid )

    torch.save(model, os.path.join(save_path,'{}_last.pth'.format(args.net)))
    # torch.save(best_model_wts, './checkpoint/{}_best.pth.pth'.format(args.net))
    print('-' * 20)
    
    #######写出最佳的权重参数########
    print('{} training finished, picture saved!'.format(args.net))
    with open ( os.path.join(save_path,"train_messages"+args.net+".txt"), "a" ) as f:
        f.write("best_score is {} epoch and validate_score is : {}".format(best_epoch + 1, best_va))

        f.write ( valid )
        
    #######开始测试########
    epoch_test_loss=0
    with open ( os.path.join(save_path,"train_messages"+args.net+".txt"), "a" ) as f:
        f.write ( "开始测试"  )
        #################加载最佳权重##################
        test_model = best_model_wts
        test_model.eval()
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = test_model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        batch_test_loss = criterion(outputs, labels)
        epoch_test_loss += batch_test_loss.item()
        outputs = F.softmax(outputs, dim=1)
        test_true.extend(labels)
        test_pred.extend(torch.argmax(outputs, dim=1))
        len_data = i+1
    epoch_test_loss = epoch_test_loss / len_data
    
    # scheduler.step(epoch_va_loss)
    test_acc = 100*accuracy_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu())
    test_f1 = f1_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu(), average='macro')
    test_recall = recall_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu(), average='macro')
    test_precision = precision_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu(), average='macro')
    # 计算平衡精度（Balanced Accuracy）
    test_bacc = balanced_accuracy_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu())
    
    with open(os.path.join(save_path, "train_messages" + args.net + ".txt"), "a") as f:
        f.write(f"test-condition----test_acc : {test_acc}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_f1 : {test_f1}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_recall : {test_recall}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_precision : {test_precision}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_bacc : {test_bacc}\n")
        f.write(valid + "\n")

import pandas as pd

# Add this snippet to the inference section where you are calculating test accuracy
def save_predictions_to_csv(test_true, test_pred,test_prob, save_path):
    # Create a DataFrame from the true and predicted labels
    test_true = torch.tensor(test_true).cpu().numpy().tolist()
    test_pred = torch.tensor(test_pred).cpu().numpy().tolist()
    test_prob = test_prob
    df = pd.DataFrame({
        'True Label': test_true,
        'Predicted Label': test_pred,
        'Test_prob': test_prob
    })
    save_path = os.path.join(save_path,"results.csv")
    # Save DataFrame to CSV
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")  
    
def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    train_loader, valid_loader,test_loader,all_loader = get_training_dataloader(args.train_dir, mean, std, input_size=args.input_size,
                                                  batch_size=args.batch_size,num_worker=args.num_workers,dying_rate = args.dying_rate,seeds = args.seed,transform_name="val")

    criterion = nn.CrossEntropyLoss()
    valid = '\n'
    save_path = f"{args.save_root}_{args.net}_{args.seed}"
    os.makedirs(save_path,exist_ok=True)
    with open ( os.path.join(save_path,"train_messages"+args.net+".txt"), "a" ) as f:
        f.write ( valid )
        f.write ( valid )
        f.write ( 'start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start-start' )
        f.write ( valid )

        
    #######开始测试########
    test_pred, test_true =[],[]
    test_prob = []
    epoch_test_loss=0
    with open ( os.path.join(save_path,"train_messages"+args.net+".txt"), "a" ) as f:
        f.write ( "开始测试"  )
        #################加载最佳权重##################
        test_model = torch.load(args.ckpt_path)
        test_model.to(device)
        test_model.eval()
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = test_model(inputs)
        # outputs = F.softmax(outputs, dim=1)
        batch_test_loss = criterion(outputs, labels)
        epoch_test_loss += batch_test_loss.item()
        outputs = F.softmax(outputs, dim=1)
        test_true.extend(labels)
        test_pred.extend(torch.argmax(outputs, dim=1))
        test_prob.extend(torch.tensor(outputs).cpu().numpy().tolist())
        len_data = i+1
    epoch_test_loss = epoch_test_loss / len_data
    
    # scheduler.step(epoch_va_loss)
    test_acc = 100*accuracy_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu())
    test_f1 = f1_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu(), average='macro')
    test_recall = recall_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu(), average='macro')
    test_precision = precision_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu(), average='macro')
    # 计算平衡精度（Balanced Accuracy）
    test_bacc = balanced_accuracy_score(torch.tensor(test_true).cpu(), torch.tensor(test_pred).cpu())
    
    save_predictions_to_csv(test_true,test_pred,test_prob,save_path)
    
    with open(os.path.join(save_path, "test_messages" + args.net + ".txt"), "a") as f:
        f.write(f"test-condition----test_acc : {test_acc}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_f1 : {test_f1}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_recall : {test_recall}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_precision : {test_precision}\n")
        f.write(valid + "\n")
        
        f.write(f"test-condition----test_bacc : {test_bacc}\n")
        f.write(valid + "\n")



if __name__ == '__main__':
    momentum = 0.5
    growth_rate = 12

    std = [0.229, 0.224, 0.225] 
    mean = [0.485, 0.456, 0.406]
    # mean = [0.5, 0.5, 0.5]
    # std = [0.2, 0.2, 0.2]
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', type=str, default='/mnt/sdb/wyz/数据/宫颈下游数据/大模型实验/泛组织ROI/', help='path of data')
    parser.add_argument('-num_classes', type=int, default=28, help='num of classes for model')
    parser.add_argument('-num_epochs', type=int, default=80, help='num of epochs for training')
    parser.add_argument('-net', type=str, default='try', help='net type')
    parser.add_argument('-device', type=str, default="cuda:0", help='use gpu or not')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-pre_trained', default=False, help='transfer learning')
    parser.add_argument('-input_size', type=int, default=224, help='input size for model')
    parser.add_argument('-num_workers', type=int, default=8, help='input size for model')
    parser.add_argument("-model_path",type=str,default="")
    parser.add_argument("-ckpt_path",type=str,default="")
    parser.add_argument("-model_name",type=str,default="virchow2")
    parser.add_argument("-mode",type=str,default="train")
    parser.add_argument("-save_root",type=str,default="/mnt/sdb/wyz/result/EXP-RESULT/2-24/")
    parser.add_argument("-dying_rate",type=int,default=1)
    
    parser.add_argument("-seed",type=int,default=2025)
    args = parser.parse_args()
    if args.mode=="train":
        train_and_valid(args)
    elif args.mode=="test":
        test(args)
