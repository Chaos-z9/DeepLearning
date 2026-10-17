from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST


# ============================================================================
# 配置结构体：所有可调参数集中在这里，改配置不用翻函数内部
# ============================================================================
@dataclass
class Config:
    # ---- 数据 ----
    data_root: str = "./data"       # Fashion-MNIST 下载/存放目录
    batch_size: int = 256           # 小批量大小
    num_workers: int = 0            # Windows 下建议 0；设 >0 需 __main__ 保护（本文件已具备）
    resize: tuple | None = None     # 如 (32, 32)；None 表示保持原图 28x28

    # ---- 模型结构：784 -> 256 -> 10 ----
    num_inputs: int = 784           # 输入维度 = 28 * 28
    num_hiddens: int = 256          # 隐藏层单元数
    num_outputs: int = 10           # 输出维度 = 类别数

    # ---- 训练 ----
    num_epochs: int = 10
    lr: float = 0.1

    # ---- 可视化与输出 ----
    figsize_curve: tuple = (10, 4)
    figsize_predict: tuple = (12, 5)
    save_dpi: int = 300
    save_curve: str = "4.2训练过程.png"
    save_predict: str = "4.2预测结果.png"

    # ---- 类别名：Fashion-MNIST 的 10 个文本标签 ----
    text_labels: tuple = (
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot',
    )


# ============================================================================
# 一、数据加载
# ============================================================================
def load_data_fashion_mnist(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """下载并加载 Fashion-MNIST，返回 (训练迭代器, 测试迭代器)。"""
    trans = [transforms.ToTensor()]
    if cfg.resize:
        trans.insert(0, transforms.Resize(cfg.resize))
    trans = transforms.Compose(trans)

    mnist_train = FashionMNIST(root=cfg.data_root, train=True,
                               transform=trans, download=True)
    mnist_test = FashionMNIST(root=cfg.data_root, train=False,
                              transform=trans, download=True)

    train_iter = DataLoader(mnist_train, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=cfg.num_workers)
    test_iter = DataLoader(mnist_test, batch_size=cfg.batch_size,
                           shuffle=False, num_workers=cfg.num_workers)
    return train_iter, test_iter


def get_fashion_mnist_labels(labels, cfg: Config) -> list[str]:
    """把数字标签转成可读的文本标签。"""
    return [cfg.text_labels[int(i)] for i in labels]


# ============================================================================
# 二、模型：单隐藏层多层感知机
# ============================================================================
def relu(X: torch.Tensor) -> torch.Tensor:
    """ReLU 激活函数：逐元素取 max(x, 0)，为网络引入非线性。"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


def init_params(cfg: Config) -> list[nn.Parameter]:
    """初始化 MLP 的全部参数，返回 [W1, b1, W2, b2]。

    权重用小方差正态分布随机初始化（打破隐藏层单元间的对称性），偏置置零。
    """
    W1 = nn.Parameter(torch.randn(cfg.num_inputs, cfg.num_hiddens) * 0.01)
    b1 = nn.Parameter(torch.zeros(cfg.num_hiddens))
    W2 = nn.Parameter(torch.randn(cfg.num_hiddens, cfg.num_outputs) * 0.01)
    b2 = nn.Parameter(torch.zeros(cfg.num_outputs))
    return [W1, b1, W2, b2]


def net(X: torch.Tensor, params: list[nn.Parameter], cfg: Config) -> torch.Tensor:
    """前向传播：展平 -> 隐藏层(仿射 + ReLU) -> 输出层(仿射)。

    注意返回的是 logits 而不是概率：交叉熵损失内部自带 softmax，
    这里若再手动 softmax 一次，既重复计算又容易数值不稳定。
    """
    W1, b1, W2, b2 = params
    X = X.reshape((-1, cfg.num_inputs))     # (B, 1, 28, 28) -> (B, 784)
    H = relu(X @ W1 + b1)                   # "@" 即矩阵乘法，H 形状 (B, 256)
    return H @ W2 + b2                      # 输出 logits，形状 (B, 10)


def make_net_fn(params: list[nn.Parameter], cfg: Config):
    """把 params 与 cfg 绑定成一个只接收 X 的前向函数，便于传给训练/评估函数。"""
    def forward(X: torch.Tensor) -> torch.Tensor:
        return net(X, params, cfg)
    return forward


# ============================================================================
# 三、评估与训练
# ============================================================================
def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """统计一批样本中预测正确的个数。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)        # 取 logits 最大的那一类作为预测
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net_fn, data_iter: DataLoader) -> float:
    """在整个数据集上评估准确率（关闭梯度以省内存、提速）。"""
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for X, y in data_iter:
            correct += accuracy(net_fn(X), y)
            total += y.numel()
    return correct / total


def train_epoch(net_fn, train_iter: DataLoader, loss_fn, updater) -> tuple[float, float]:
    """训练一个 epoch，返回 (平均损失, 平均准确率)。"""
    loss_sum, correct, total = 0.0, 0.0, 0.0
    for X, y in train_iter:
        y_hat = net_fn(X)                   # 前向：算预测
        l = loss_fn(y_hat, y)               # 算损失，形状 (batch_size,)
        updater.zero_grad()                 # 清空上一轮残留梯度
        l.mean().backward()                 # 反向：求各参数梯度
        updater.step()                      # 沿负梯度方向更新参数
        loss_sum += float(l.detach().sum())   # 先 detach 再取值，避免 requires_grad 警告
        correct += accuracy(y_hat, y)
        total += y.numel()
    return loss_sum / total, correct / total


def train(net_fn, train_iter: DataLoader, test_iter: DataLoader,
          loss_fn, updater, cfg: Config) -> tuple[list[float], list[float], list[float]]:
    """完整训练流程，返回三个历史记录：训练损失、训练准确率、测试准确率。"""
    loss_history: list[float] = []
    train_acc_history: list[float] = []
    test_acc_history: list[float] = []

    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_epoch(net_fn, train_iter, loss_fn, updater)
        test_acc = evaluate_accuracy(net_fn, test_iter)

        loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(f'epoch {epoch + 1:2d}/{cfg.num_epochs}  '
              f'loss {train_loss:.4f}  train acc {train_acc:.4f}  test acc {test_acc:.4f}')

    return loss_history, train_acc_history, test_acc_history


# ============================================================================
# 四、可视化
# ============================================================================
def plot_metrics(loss_history: list[float], train_acc_history: list[float],
                 test_acc_history: list[float], cfg: Config) -> None:
    """左图 loss 曲线，右图 train/test 准确率曲线；先保存再显示。"""
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=cfg.figsize_curve)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, 'r-o', label='Train Acc')
    plt.plot(epochs, test_acc_history, 'g-s', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(cfg.save_curve, dpi=cfg.save_dpi)   # 保存必须放在 show 之前
    plt.show()


def predict(net_fn, test_iter: DataLoader, cfg: Config, n: int = 10) -> None:
    """取一个 batch 做预测，画成 2x5 网格对比真实/预测标签；先保存再显示。"""
    X, y = next(iter(test_iter))
    preds = net_fn(X).argmax(dim=1)
    true_labels = get_fashion_mnist_labels(y, cfg)
    pred_labels = get_fashion_mnist_labels(preds, cfg)

    rows, cols = 2, (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=cfg.figsize_predict)
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].squeeze(), cmap='gray')      # squeeze 去掉通道维，得到 H x W
        hit = true_labels[i] == pred_labels[i]
        ax.set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}',
                     color='green' if hit else 'red')
        ax.axis('off')

    plt.suptitle("Prediction Results (Green=Correct, Red=Wrong)")
    plt.tight_layout()
    plt.savefig(cfg.save_predict, dpi=cfg.save_dpi)  # 保存必须放在 show 之前
    plt.show()


# ============================================================================
# 五、主函数
# ============================================================================
def main() -> None:
    cfg = Config()

    # 1) 准备数据
    train_iter, test_iter = load_data_fashion_mnist(cfg)

    # 2) 初始化模型参数，并绑定出只接收 X 的前向函数
    params = init_params(cfg)
    net_fn = make_net_fn(params, cfg)

    # 3) 损失函数与优化器
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(params, lr=cfg.lr)

    # 4) 训练
    loss_history, train_acc_history, test_acc_history = train(
        net_fn, train_iter, test_iter, loss_fn, updater, cfg)

    # 5) 可视化训练过程与预测结果
    plot_metrics(loss_history, train_acc_history, test_acc_history, cfg)
    predict(net_fn, test_iter, cfg)


if __name__ == '__main__':
    main()
