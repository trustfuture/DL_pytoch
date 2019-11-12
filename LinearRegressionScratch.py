import torch
import  numpy as np
import random
# from matplotlib import pyplot as plt

# 生成数据集
n_examples = 1000
n_inputs = 2
true_w = [3, -1.2]
true_b = 2.5
features = torch.rand(n_examples, n_inputs)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (n_inputs, 1)), dtype=torch.float)
b = torch.zeros(1)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型
def linear_regression(X, w, b):
    return torch.mm(X, w) + b # matrix multiplication

# 定义损失函数
def square_loss(y, y_hat):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

# 读取数据
def data_iter(batch_size, features, labels):
    n_examples = len(features)
    indexs = list(range(n_examples))
    random.shuffle(indexs)
    for i in range(0, n_examples, batch_size):
        j = torch.LongTensor(indexs[i: min(i+batch_size, n_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)



# 训练模型
batch_size = 10
lr = 0.03
n_epoch = 30

for epoch in range(n_epoch):
    train_loss = 0.0
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linear_regression(X, w, b)
        loss = square_loss(y, y_hat).sum() # 小批量样本的损失和
        train_loss += loss
        loss.backward() # 计算梯度
        sgd([w,b], lr, batch_size)

        # 更新完一次参数，把梯度清理，否则梯度是累加的
        w.grad.data.zero_()
        b.grad.data.zero_()
    mean_loss = train_loss / n_examples
    print('epoch %d, loss %f' % (epoch + 1, mean_loss.item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
