import torch
from torch import nn
import numpy as np

# 生成数据集
n_examples = 1000
n_inputs = 2
true_w = [3, -1.2]
true_b = 2.5
features = torch.rand(n_examples, n_inputs)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据
batch_size = 10
import torch.utils.data as Data
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

# 定义模型
net = nn.Sequential(nn.Linear(n_inputs, 1))

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
from torch import optim
optimizer = optim.SGD(net.parameters(),lr=0.03)

# 初始化参数
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

#训练过程
num_epoch = 10
for epoch in range(num_epoch):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch + 1, l.item()))

dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)