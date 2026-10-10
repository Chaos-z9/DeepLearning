import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ===================== 1. 获取数据集 =====================
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size, resize=None):
    """下载并加载Fashion-MNIST数据集，返回DataLoader迭代器"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = FashionMNIST(root="./data", train=False, transform=trans, download=True)

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter



# ===================== 3. 定义Softmax运算 =====================
def softmax(X):
    """手动实现softmax，避免数值溢出"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


# ===================== 4. 定义模型 =====================
def net(X):
    """线性分类模型 + softmax"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# ===================== 5. 定义损失函数（交叉熵） =====================
def cross_entropy(y_hat, y):
    """手动实现交叉熵损失"""
    return -torch.log(y_hat[range(len(y_hat)), y])


# ===================== 6. 定义准确率计算 =====================
def accuracy(y_hat, y):
    """计算预测正确的样本数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在多个变量上累加值"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(data_iter, net_fn):
    """计算在指定数据集上的准确率"""
    metric = Accumulator(2)  # [正确预测数, 总预测数]
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net_fn(X), y), y.numel())
    return metric[0] / metric[1]


# ===================== 7. 训练函数 =====================
def train_epoch_ch3(net_fn, train_iter, loss_fn, updater):
    """训练一个epoch"""
    metric = Accumulator(3)  # [训练损失总和, 训练准确率总和, 样本数]
    for X, y in train_iter:
        y_hat = net_fn(X)
        l = loss_fn(y_hat, y)
        # 反向传播 & 更新参数
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 自定义SGD
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# ============================================================================
if __name__ == '__main__':

    # ===================== 1. 获取数据集 =====================
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # ===================== 2. 初始化模型参数 =====================
    num_inputs = 784   # 28x28 展平
    num_outputs = 10   # 10个类别

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # ===================== 8. 主训练循环 =====================
    num_epochs = 10
    lr = 0.1

    loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, cross_entropy,
                                                lambda bs: sgd([W, b], lr, bs))
        test_acc = evaluate_accuracy(test_iter, net)

        loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(f'Epoch {epoch+1}: loss={train_loss:.4f}, '
              f'train_acc={train_acc:.4f}, test_acc={test_acc:.4f}')

    # ===================== 9. 可视化训练过程 =====================
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), loss_history, 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_acc_history, 'r-o', label='Train Acc')
    plt.plot(range(1, num_epochs+1), test_acc_history, 'g-s', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('3.6训练过程.png', dpi=300)  #保存图片在show之前
    plt.show()

    # ===================== 10. 预测示例 =====================
    X, y = next(iter(test_iter))
    preds = net(X).argmax(dim=1)
    true_labels = get_fashion_mnist_labels(y)
    pred_labels = get_fashion_mnist_labels(preds)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img = X[i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}', color=color)
        ax.axis('off')
    plt.suptitle("Prediction Results (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.savefig('3.6预测结果.png', dpi=300)  #保存图片在show之前
    plt.show()