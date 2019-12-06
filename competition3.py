# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:43:28 2019

@author: Lenovo
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from PIL import Image


#处理训练集数据
transform = T.Compose([
    T.ToTensor(),  
])

class LoadTrain(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transform   
        
    def code_lable(self, label):
        lable_cat = torch.Tensor([])
        for i in range(len(label)):
            num = ord(label[i])-48
            if num>9:
                num -= 7
                if num>35:
                    num -= 6
                
            arr = torch.zeros(1, 62)
            arr[:,num] = 1
            lable_cat = torch.cat((lable_cat,arr),dim=1)
        return lable_cat
        
    def __getitem__(self, index):
        image_path = self.paths[index]    
        label = list(image_path)[:-4]
        label = self.code_label(label).reshape(310)
        
        if image_path !='.DS_Store':
            pil_image = Image.open(self.root+'\\'+image_path)
            if self.transforms:
                data = self.transforms(pil_image)
            else:
                image_array = np.asarray(pil_image)
                data = torch.from_numpy(image_array)
        return data, label

    def __len__(self):
        return len(self.paths)
    
#处理测试集图片数据   
class LoadTest(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transform   
        
    def __getitem__(self, index):
        image_path = self.paths[index]
        
        if image_path !='.DS_Store':
            pil_image = Image.open(self.root+'\\'+image_path)   
            if self.transforms:
                data = self.transforms(pil_image)
            else:
                image_array = np.asarray(pil_image)
                data = torch.from_numpy(image_array)
            return data 
        
    def __len__(self):
        return len(self.paths)
  
#构建残差块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            #nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#构建残差网络
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=310):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.fc = nn.Linear(4608, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#将残差块代入神经网络
def ResNets():
    return ResNet(ResidualBlock)


#训练模型
def training():
    data = LoadTrain('train/train', transform)
    dataloader = DataLoader(data, batch_size=32, shuffle=True, drop_last=False)
    img, label = data[0]
    print(img.shape)
    
    cnn = ResNets()
    
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters())
    
    for i in range(1):  
        for j,(img,labels) in enumerate(dataloader):
            out = cnn(img)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                print('i=%d j=%d Loss: %.5f' %(i,j,loss.item()))
    
    torch.save(cnn.state_dict(),'can3.pt')
    
    return

#ASCII码转化为字符
def uncode(code):
    tabel = []
    for i in range(len(code)):
        if code[i]<10:
            tabel.append(chr(code[i]+48))
        elif 10<=code[i]<36:
            tabel.append(chr(code[i]+55))
        else: 
            tabel.append(chr(code[i]+61))
    return tabel

#预测测试集图片类别
def main():
    data = LoadTest('test/test', transform)
    cnn = ResNets()
    cnn.load_state_dict(torch.load('can3.pt'))
    cnn.eval()

    prediction = []
    
    for i in range(len(data)):
        if i %1000 == 0:
            print(i)
        imgs = data[i]
        imgs = torch.Tensor(imgs).reshape(1,3,30,150)
        output = cnn(imgs)
        output = output.view(-1, 62)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        out = uncode(output)
        print(out)
        prediction.append(out)
        
    m = len(prediction)
    labels = [[] for i in range(m)]
    for i in range(m):  
            labels[i].append(i)
            labels[i].append(prediction[i])
            
    np.savetxt('sample_data2.csv',labels, fmt='%s')
    
#调用定义好的函数完成预测
training()
main()
    
