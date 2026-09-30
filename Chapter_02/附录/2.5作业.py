import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(-2 * np.pi, 2 * np.pi, 200, requires_grad=True)
y=torch.sin(x)
x.requires_grad_(True)
#x.grad.zero_()
y.sum().backward()
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
plt.savefig('sin_derivative.png')
plt.show()



