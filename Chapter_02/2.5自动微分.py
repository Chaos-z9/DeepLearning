import torch

x = torch.arange(4.0)
print(x)
#tensor([0., 1., 2., 3.])
x.requires_grad_(True)
print(x.grad)
#None
y=2*torch.dot(x,x)
print(y)
#tensor(28., grad_fn=<MulBackward0>)

y.backward()
print(x.grad)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)


#2.5.3. 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u




