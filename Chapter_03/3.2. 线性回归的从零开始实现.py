import random
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus']=False

def synthetic_data(w, b, num_examples): 
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))#形状是 (num_examples, len(w))，即 (1000, 2)
    y = torch.matmul(X, w) + b  #torch.matmul 是矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


print('features:', features[0],'\nlabel:', labels[0])
def plot_eda():
    plt.figure()
    plt.scatter(features[:,1].numpy(), labels.numpy(), s=50, c='blue',marker='o',label='实验点')
    plt.legend(['实验点'])
    plt.title('合成数据集EDA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True,alpha=0.5)#alpha为透明度属性
    #plt.savefig('合成数据集EDA.png',dpi=300)  #保存图片在show之前
    plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))#样本索引
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # 从打乱后的索引列表里，切出当前这一批的索引。
        #为什么要有 min(i + batch_size, num_examples)？防止越界！
        yield features[batch_indices], labels[batch_indices]
        #根据 batch_indices 里的编号，去 features 和 labels 里把对应的数据抠出来，扔给训练循环。
        #关于 yield 的作用（可以看成return）：它一次只在内存里保留一个 batch 的数据，绝不把整个数据集一次性塞进内存！ 


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)



def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y): 
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def plot_loss(epoch_losses):
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='blue')
    plt.title('训练损失随迭代次数的变化')
    plt.xlabel('迭代次数Epoch')
    plt.ylabel('训练损失Loss')
    plt.grid(True,alpha=0.5)
    plt.legend(['训练损失'])
    #plt.savefig('Loss曲线.png',dpi=300)  #保存图片在show之前
    plt.show()


def train():
    lr = 0.03
    num_epochs = 3
    #函数指针指向模型和损失函数
    net = linreg
    loss = squared_loss
    epoch_losses = []  # 记录每个epoch的损失
    for epoch in range(num_epochs):
        #分批次（Batch） 
        for X, y in data_iter(batch_size, features, labels):
            #算误差（Forward）
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            #求导（Backward）
            l.sum().backward()
            #沿着梯度反方向挪一步（SGD）
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        #清空旧梯度
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            current_loss = float(train_l.mean())
            epoch_losses.append(current_loss) 
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    #绘制loss曲线
    plot_loss(epoch_losses)


if __name__ == '__main__':
    plot_eda()
    train()




    