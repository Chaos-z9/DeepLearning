# 图像分类 Softmax 完整流程笔记

> 说明：笔记思路：以Softmax核心知识为主线，中文语句作为思路引导，`Q:`标记个人存疑点位；

## ① 模型本体

**解决问题：图片分类**

图片 $\xrightarrow{\text{信息论}}$ 张量

> Q：具体是什么？如何转变的？

模型核心：线性变换

$$
\begin{cases}
o_1 = \mathbf{w}_1^\top \mathbf{x} + b_1 \\
o_2 = \mathbf{w}_2^\top \mathbf{x} + b_2 \\
o_3 = \mathbf{w}_3^\top \mathbf{x} + b_3
\end{cases}
$$

> Q：线性变换为什么可以分类图片？有什么理论依据？

$o_1, o_2, o_3$ 是打分值（logits），我们希望模型输出对准预测类别，需要将打分转化为概率，引入 softmax 函数做非线性校准：

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})
$$

$$
\hat{y}_j=\frac{e^{o_j}}{\sum_{k=1}^q e^{o_k}}=\frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)}
$$

## ② 损失

问题：如何衡量 “概率” 的好坏？“概率” 本身不可直接优化，因此我们需要建立预测概率与真实标签之间的关系。

$$
P(Y \mid X)=\prod_{i=1}^n P(y^{(i)} \mid x^{(i)})
$$

这本质是条件概率。我们最大化 $P(Y \mid X)$，等价于最小化负对数似然：

$$
-\log P(Y \mid X)=\sum_{i=1}^n -\log P(y^{(i)} \mid x^{(i)})=\sum_{i=1}^n l(y^{(i)},\hat{y}^{(i)})
$$

## ③ 反向传播

问题：有没有比较奇怪？在线性网络中 $\hat{y} = \mathbf{w}^\top\mathbf{x}+b$，单层的梯度相对容易理解，但是在全连接层接上 softmax 之后梯度推导理解起来会复杂很多。

> Softmax 导数:
> 将 softmax 定义代入交叉熵损失，再对未规范化的预测 $o_j$ 求导：

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &= -\sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

可见损失对 logits 的导数就是「预测概率减去真实标签」，再沿 $\mathbf{o}=\mathbf{W}\mathbf{x}+\mathbf{b}$ 回传，即可求得 $\frac{\partial l}{\partial \mathbf{W}},\frac{\partial l}{\partial \mathbf{b}}$。

## ④ 优化器（Mini‑batch SGD）

$$
\mathbf{W} \leftarrow \mathbf{W}-\frac{\eta}{|\mathcal{B}|}\nabla_{\mathbf{W}} l
$$

$$
\mathbf{b} \leftarrow \mathbf{b}-\frac{\eta}{|\mathcal{B}|}\nabla_{\mathbf{b}} l
$$
