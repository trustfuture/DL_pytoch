import torch
from torch import nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义参数
num_inputs, num_hiddens, num_outputs = 784, 256, 10
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# 定义激活函数
def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))

# 定义模型
def net(X):
    X = X.view(-1, num_inputs) # X shape (batch_size, 1, 28, 28) reshape (-1, 784)
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 训练过程
'''
注：由于原书的mxnet中的SoftmaxCrossEntropyLoss在反向传播的时候相对于沿batch维求和了，
而PyTorch默认的是求平均，所以用PyTorch计算得到的loss比mxnet小很多（大概是maxnet计算得到的1/batch_size这个量级），
所以反向传播得到的梯度也小很多，所以为了得到差不多的学习效果，我们把学习率调得成原书的约batch_size倍，原书的学习率为0.5，
这里设置成100.0。(之所以这么大，应该是因为d2lzh_pytorch里面的sgd函数在更新的时候除以了batch_size，
其实PyTorch在计算loss的时候已经除过一次了，sgd这里应该不用除了)
'''
num_epochs = 5
lr =100.0

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)