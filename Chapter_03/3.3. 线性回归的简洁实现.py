import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils import data
from torch import nn
from dataclasses import dataclass
#我们希望用工程化的角度来写代码

plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus']=False

@dataclass
class Config:
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    num_epochs = 50
    batch_size = 10
    epoch_losses = []

def synthetic_data(w, b, num_examples): 
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))#形状是 (num_examples, len(w))，即 (1000, 2)
    y = torch.matmul(X, w) + b  #torch.matmul 是矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


net = nn.Sequential(nn.Linear(2, 1))
def Init():
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

def train(net, loss, trainer, num_epochs,data_iter):       
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        Config.epoch_losses.append(float(l))
        print(f'epoch {epoch + 1}, loss {l:f}')

def plot_loss(epoch_losses):
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='blue')
    plt.title('训练损失随迭代次数的变化')
    plt.xlabel('迭代次数Epoch')
    plt.ylabel('训练损失Loss')
    plt.grid(True,alpha=0.5)
    plt.legend(['训练损失'])
    plt.savefig('3.31Loss曲线.png',dpi=300)  #保存图片在show之前
    plt.show()

if __name__ == '__main__':
    features, labels = synthetic_data(Config.true_w, Config.true_b, 1000)
    data_iter = load_array((features, labels), batch_size=10)
    train(net, loss, trainer, Config.num_epochs,data_iter)
    plot_loss(Config.epoch_losses)







