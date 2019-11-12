import d2lzh_pytorch as d2l
import torch
import torchvision
import numpy as np
import sys

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化参数
num_inputs = 784
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型

## 实现softmax
def softmax(mat): # 输入形状是num_examples * num_outputs
    mat_exp = mat.exp()
    partition = mat_exp.sum(dim=1, keepdim=True)
    return mat_exp / partition

## 实现模型
def net(X):
    return softmax(torch.mm(X.view(-1, num_inputs), w) + b)

# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# 定义优化算法
# 使用之前写过的 放在util里的sgd

# 定义评估函数
## 一次数据的精确度
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

## 在数据集上的精确度
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# 训练过程
num_epochs = 5
lr = 0.1

# 个人简洁版，不包括optimizer部分，完整版见utils.py里的train_ch3函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_acc_sum, train_loss_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            loss = cross_entropy(y_hat, y).sum()
            loss.backward()
            d2l.sgd(params, lr, batch_size)
            for param in params:
                param.grad.data.zero_()
            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)

# 预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])