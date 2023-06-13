'''
Author: huzihe06@gmail.com
Date: 2023-02-26 21:49:36
LastEditTime: 2023-03-11 12:02:40
FilePath: \pyplot\TensorFlow\practice\cnn-gnss.py
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
torch.__version__

class tabularDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

class gnss1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 6 * 6)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

class gnssModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, 2)
        self.bn_in = nn.BatchNorm1d(4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)


    def forward(self,x_in):
        x = self.bn_in(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)

        x = F.relu(self.lin2(x))
        x = self.bn2(x)

        x = self.lin3(x)
        x=torch.sigmoid(x)
        return x

#两层卷积层，后面接一个全连接层
class gnssLearn(nn.Module):
    def __init__(self):
        super(gnssLearn, self).__init__()
        self.model1 = nn.Sequential(
        	#输入通道一定为1，输出通道为卷积核的个数，2为卷积核的大小（实际为一个[1,2]大小的卷积核）
            nn.Conv1d(1, 16, 2),  
            nn.Sigmoid(),
            nn.MaxPool1d(2),  # 输出大小：torch.Size([128, 16, 5])
            nn.Conv1d(16, 32, 2),
            nn.Sigmoid(),
            nn.MaxPool1d(1),  # 输出大小：torch.Size([128, 32, 1])
            nn.Flatten(),  # 输出大小：torch.Size([128, 32])
        )
        self.model2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=2, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, input):
        x = self.model1(input)
        x = self.model2(x)
        return x


class tabularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 500)
        self.lin2 = nn.Linear(500, 100)
        self.lin3 = nn.Linear(100, 2)
        self.bn_in = nn.BatchNorm1d(4)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(100)


    def forward(self,x_in):
        #print(x_in.shape)
        x = self.bn_in(x_in)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        #print(x)


        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        #print(x)

        x = self.lin3(x)
        x=torch.sigmoid(x)
        return x

path = "./data/ml-data/gnss-data-20230129-1.csv"
gnssdata = pd.read_csv(path)
print(gnssdata.shape)
print(gnssdata.describe())
# print(gnssdata.isnull().any())
x = gnssdata.drop(["los"], axis=1)
y = gnssdata["los"]

train_ds = tabularDataset(x, y)

# a= train_ds[0]

#训练前指定使用的设备
DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda")
print(DEVICE)

#损失函数
criterion =nn.CrossEntropyLoss()

#实例化模型
# model = tabularModel().to(DEVICE)
# model = gnssModel().to(DEVICE)
model =gnssLearn().to(DEVICE)
print(model)

# #测试模型是否没问题
# rn=torch.rand(2,4).to(DEVICE)
# model(rn)

#学习率
LEARNING_RATE=0.01
#BS
batch_size = 4
#优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#DataLoader加载数据
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)

# %%time
model.train()
#训练10轮
TOTAL_EPOCHS=10
#记录损失函数
losses = [];
for epoch in range(TOTAL_EPOCHS): 
    for i, (x, y) in enumerate(train_dl):
        x = x.float().to(DEVICE) #输入必须为float类型
        y = y.long().to(DEVICE) #结果标签必须为long类型
        #清零
        optimizer.zero_grad()
        outputs = model(x)
        #计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, np.mean(losses)))

model.eval()
correct = 0
total = 0
for i,(x, y) in enumerate(train_dl):
    x = x.float().to(DEVICE)
    y = y.long()
    outputs = model(x).cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
print('准确率: %.4f %%' % (100 * correct / total))