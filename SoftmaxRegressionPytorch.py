import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义模型

## 定义线性层
num_inputs = 784
num_outputs = 10

# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#
#     def forward(self, x): # x shape (batch_size, 1, 28, 28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return y

# net = LinearNet(num_inputs, num_outputs)

## 定义一个展开层，方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape (batch_size, *, *, ...）
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_outputs)
)

# 初始化参数
init.normal_(net[1].weight, mean=0, std=0.01)
init.constant_(net[1].bias, val=0)

# Softmax和交叉熵损失函数，torch里合在一起了
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型，用前面定义过的函数训练
num_epoch = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, batch_size, None, None, optimizer)

