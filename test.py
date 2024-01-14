import torch



import numpy as np



import logging

from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from dataloader.dataloader import Dataset, DataLoader
from loss.loss import mseLoss
from trainer.trainer import Trainer

logging.basicConfig(level=logging.INFO)


# 定义线性层
m = nn.Sequential(
    nn.Linear(2, 1000, dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(1000, 1, dtype=torch.float64)
)

m.to("cuda:3")

# 随机100*2大小的数据服从标准正态分布
data = np.random.randn(100, 2)

# 生成数据对应的标签
labels = data[:, 0:1] + 10 * data[:, 1:2]
class_labels = [1 if data[i][j] > 5 else -1 for j in range(data.shape[1]) for i in range(data.shape[0]) ]

# 生成数据集
dataset = Dataset(data, labels)

# 分割训练集 和 测试集
train_dataset, test_dataset = dataset.split(0.8)

# 优化器
opt = SGD(params=m.parameters(), lr=0.000001)

sch = None

trainer = Trainer(m, opt, mseLoss, {
    "epochs": 100,
    "batch_size": 10,
    "shuffle": False,
    "save_path": "./tmp/model.pkl"
}, sch)

# 计算损失
train_dataloader = DataLoader(train_dataset, trainer.batch_size, shuffle=trainer.shuffle)
test_dataloader = DataLoader(test_dataset, trainer.batch_size, shuffle=trainer.shuffle)
loss = trainer.train(train_dataloader, test_dataloader)

print(loss)
