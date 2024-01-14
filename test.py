import torch

from lightGE.core.nn import Model, Sequential, Linear, Conv2d
from lightGE.data import Dataset, DataLoader
from lightGE.utils import mseLoss, maeLoss, crossEntropyLoss, hingeLoss, multi_classification_kld, mbeLoss, rmseLoss
from lightGE.core.nn import Linear
from lightGE.utils.scheduler import MultiStepLR, StepLR, Exponential, Cosine

import numpy as np

from lightGE.utils import SGD, Trainer

import logging

logging.basicConfig(level=logging.INFO)

# 定义线性层
m = Linear(2, 1)

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
opt = SGD(parameters=m.parameters(), lr=0.01)

sch = MultiStepLR(opt, [10, 20, 30, 40, 50, 60, 70, 80, 90])

trainer = Trainer(m, opt, mseLoss, {
    "epochs": 100,
    "batch_size": 10,
    "shuffle": True,
    "save_path": "./tmp/model.pkl"
}, sch)

# 计算损失
train_dataloader = DataLoader(train_dataset, trainer.batch_size, shuffle=trainer.shuffle)
test_dataloader = DataLoader(test_dataset, trainer.batch_size, shuffle=trainer.shuffle)
loss = trainer.train(train_dataloader, test_dataloader)

print(loss)
