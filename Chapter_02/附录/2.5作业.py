import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(-2 * np.pi, 2 * np.pi, 200, requires_grad=True)
y=torch.sin(x)
x.requires_grad_(True)
#x.grad.zero_()      这里注意！没有梯度相当于释放空指针
y.sum().backward()   #这里backward()一定是一维张量
print(x.grad)

x_data = x.detach().numpy()
y_data = y.detach().numpy()
grad_data = x.grad.numpy() 

plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, label='f(x) = sin(x)', color='blue')
plt.plot(x_data, grad_data, label="f'(x) - computed by autograd", color='red', linestyle='--')

# 顺便把真实导数（cos(x)）也画上，看看对不对得上
true_grad = np.cos(x_data)
plt.plot(x_data, true_grad, label="true f'(x) = cos(x)", color='green', linestyle=':')
plt.title("Derivative of sin(x)")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
#plt.savefig('sin_derivative.png')
plt.show()




#在运行反向传播函数之后，立即再次运行它，看看会发生什么？

x = torch.arange(4.0, requires_grad=True)
print("初始 x.grad:", x.grad)  # None (空指针)

# 第一次前向 + 反向
y = 2 * torch.dot(x, x)
y.backward()
print("第一次 backward 后的 x.grad:", x.grad)  # tensor([ 0.,  4.,  8., 12.])

# 【作死尝试1】直接第二次 backward，不加 retain_graph
try:
    y.backward()
except RuntimeError as e:
    print("\n报错了！错误信息：", e)

# 【正确尝试2】重新前向传播，再 backward
y = 2 * torch.dot(x, x)  # 重新建图
y.backward()
print("\n第二次重新建图后，x.grad:", x.grad)  # tensor([ 0.,  8., 16., 24.]) —— 翻倍了！

# 【正确尝试3】手动清零，看看效果
x.grad.zero_()  # 相当于 C++ 的 memset
print("\n清零后 x.grad:", x.grad)

y = 2 * torch.dot(x, x)
y.backward()
print("重新建图并清零后 x.grad:", x.grad)  # 又回到 [ 0.,  4.,  8., 12.]


