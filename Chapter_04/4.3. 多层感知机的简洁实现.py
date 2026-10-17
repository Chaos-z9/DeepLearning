from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import torch
from torch import nn

#我们希望用工程化的角度来写代码
#我们来试一下全新的简洁实现
#本节做三组对照实验：对照组 MLP-256、实验组一 MLP-4、实验组二 线性回归

plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus']=False


# ============================================================================
# 对照组描述：一个组 = 一份模型规格 + 一份训练结果
# ============================================================================
@dataclass
class GroupSpec:
    """单个对照组的定义与结果。"""
    name: str                       # 组名（图例里显示）
    hidden: int                     # 隐藏层神经元数；0 表示纯线性回归（无隐藏层）
    color: str                      # 拟合曲线颜色
    linewidth: float = 2.5          # 线宽
    linestyle: str = '-'            # 线型
    losses: list = field(default_factory=list)  # 每轮损失记录
    model: object = None                        # 训练好的模型
    final_loss: float = 0.0                     # 最后一个 epoch 的损失


# ============================================================================
# 全局配置
# ============================================================================
@dataclass
class Config:
    # ---- 模型结构：1 -> hidden -> 1 ----
    num_inputs: int = 1             # 输入维度 = 1（x）
    num_outputs: int = 1            # 输出维度 = 1（预测的 y）
    # ---- 对照实验分组：名字直接区分对照组 / 实验组 ----
    hidden_control: int = 256       # 对照组：隐藏层 256 个神经元（基准）
    hidden_exp1: int = 4            # 实验组一：隐藏层 4 个神经元（砍容量）
    hidden_exp2: int = 0            # 实验组二：隐藏层 0 个，即纯线性回归
    # ---- 训练 ----
    num_epochs: int = 2000          # sin 拟合需要多轮
    lr: float = 0.01
    # ---- 可视化与输出 ----
    figsize_fitting: tuple = (11, 6)
    save_dpi: int = 300
    save_fitting: str = "4.3三组对照拟合.png"
    # ---- 数据 ----
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 1000).unsqueeze(1) # 形状 (1000, 1)
    y = torch.sin(x) + 0.1 * torch.randn(x.shape)
    # ---- 三组对照清单（在 __post_init__ 里按上面的配置装配，保证只有一个来源）----
    groups: list = field(default_factory=list)

    def __post_init__(self):
        self.groups = [
            GroupSpec('对照组   MLP-256',  self.hidden_control, 'red',    3.0, '-'),
            GroupSpec('实验组一 MLP-4',    self.hidden_exp1,    'orange', 2.5, '-'),
            GroupSpec('实验组二 线性回归',  self.hidden_exp2,    'blue',   2.0, '--'),
        ]


# ============================================================================
# 模型
# ============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # 输入层 -> 隐藏层
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # 激活函数：非线性折弯
        self.relu = nn.ReLU()
        # 隐藏层 -> 输出层
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.layer1(x) # 线性变换
        h = self.relu(h)   # 非线性激活
        out = self.layer2(h)
        return out


def build_model(hidden: int) -> nn.Module:
    """按隐藏层宽度造模型。

    hidden > 0 : 单隐藏层 MLP（Linear -> ReLU -> Linear）
    hidden == 0: 纯线性回归，数学上就是 y = wx + b，用 nn.Linear(1, 1) 一行搞定
    """
    if hidden > 0:
        return MLP(1, hidden, 1)
    return nn.Linear(1, 1)


# ============================================================================
# 训练
# ============================================================================
def train(model, cfg: Config, criterion, optimizer, verbose: bool = True) -> list[float]:
    """训练一个模型，返回每轮的损失列表。

    cfg 作为形参显式传入，函数内部不依赖全局变量。
    """
    losses: list[float] = []
    for epoch in range(cfg.num_epochs):
        y_pred = model(cfg.x)                # 前向传播
        loss = criterion(y_pred, cfg.y)      # 算误差

        optimizer.zero_grad()                # 清空梯度
        loss.backward()                      # 反向传播
        optimizer.step()                     # 更新参数

        losses.append(loss.item())           # 每轮都记录，曲线才连续
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")

    return losses


def run_group(group: GroupSpec, cfg: Config, criterion) -> None:
    """跑一个对照组：建模型 -> 训练 -> 把结果写回 group。"""
    print(f'--- {group.name}  (hidden={group.hidden}) ---')
    group.model = build_model(group.hidden)
    optimizer = torch.optim.Adam(group.model.parameters(), lr=cfg.lr)
    group.losses = train(group.model, cfg, criterion, optimizer)
    group.final_loss = group.losses[-1]
    print(f'  => 最终损失 {group.final_loss:.6f}\n')


# ============================================================================
# 可视化：三组画在同一张图上对照
# ============================================================================
def plot_comparison(cfg: Config) -> None:
    """把所有对照组的拟合曲线画在同一张图上，便于直接比较。"""
    plt.figure(figsize=cfg.figsize_fitting)

    # 先画背景元素，保证拟合曲线压在上面
    plt.scatter(cfg.x.numpy(), cfg.y.numpy(), s=2, color='gray', alpha=0.3,
                label='真实带噪数据', zorder=1)
    plt.plot(cfg.x.numpy(), torch.sin(cfg.x).numpy(), color='green',
             linewidth=2, label='真实规律 sin(x)', zorder=2)

    # 逐个对照组画拟合曲线，图例里带上各自的最终损失
    for g in cfg.groups:
        g.model.eval()                       # 切到评估模式
        with torch.no_grad():
            y_fit = g.model(cfg.x).numpy()
        plt.plot(cfg.x.numpy(), y_fit, color=g.color, linewidth=g.linewidth,
                 linestyle=g.linestyle, zorder=3,
                 label=f'{g.name} (loss={g.final_loss:.4f})')

    plt.title('三组对照：MLP-256 / MLP-4 / 线性回归 拟合 sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(cfg.save_fitting, dpi=cfg.save_dpi) # 保存图片在show之前
    plt.show()
    print(f'图已保存：{cfg.save_fitting}')


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    cfg = Config()
    criterion = nn.MSELoss()

    # 1) 依次跑三组对照实验
    for g in cfg.groups:
        run_group(g, cfg, criterion)

    # 2) 打印汇总表
    print('======== 汇总 ========')
    for g in cfg.groups:
        print(f'{g.name}  hidden={g.hidden:<4}  最终损失={g.final_loss:.6f}')

    # 3) 三组拟合曲线画在同一张图上对照
    plot_comparison(cfg)


if __name__ == "__main__":
    main()
