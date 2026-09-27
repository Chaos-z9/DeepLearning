import torch

#2.1.1 张量的基础知识

x = torch.arange(12)
print(x)
#tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

print(x.shape)
#torch.Size([12])

print(x.numel())
#12

X=x.reshape((3,4))
print(X)
#tensor([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]]])

print(torch.zeros((2,3,4)))
# (2,3,4)：指定张量的形状（shape），表示：
# 第0维：2个元素
# 第1维：3个元素
# 第2维：4个元素
# 总共：2×3×4 = 24个元素
# 同理: torch.ones((2,3,4))，生成全1的张量，形状为(2,3,4)  


print(torch.randn(3,4))
# 生成3行4列的随机张量，元素服从标准正态分布（均值为0，标准差为1）。通常用于随机初始化参数

print(torch.tensor([[1,2,3],[4,5,6]]))
#生成2行3列的张量，元素为1,2,3,4,5,6


#2.1.2 张量的操作

#基本原则：按元素运算
# (+,-,*,/,**,torch.exp(x))

#张量的拼接
X=torch.arange(12,dtype=torch.float32).reshape(3,4)
# 创建一个3×4的张量X
# torch.arange(12)：生成0到11的序列
# dtype=torch.float32：指定数据类型为32位浮点数
# .reshape(3,4)：重塑为3行4列的矩阵

Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
#直接创建一个3×4的张量Y，值为给定的二维列表

print(torch.cat((X,Y),dim=0))
# torch.cat((X,Y), dim=0) —— 按行拼接（沿第0维）
# 将X和Y在行方向上堆叠
# 结果形状：(6, 4)，即6行4列
# X的3行 + Y的3行 = 6行
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])

print(torch.cat((X,Y),dim=1))
# torch.cat((X,Y), dim=1) —— 按列拼接（沿第1维）
# 将X和Y在列方向上拼接
# 结果形状：(3, 8)，即3行8列
# X的4列 + Y的4列 = 8列
# tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])

#用逻辑运算符构建张量
print(X==Y)

#对所有元素求和，会产生单一元素张量
print(X.sum())


#2.1.3 广播机制
# 广播机制：当两个张量的形状不同时，通过自动扩展（broadcast）来匹配形状，实现按元素运算
# 在大多数运算下，我们将沿着数组长度为1的维度进行广播，以匹配另一个张量的形状
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
print(a)
print(b)
print(a+b)


#2.1.4 索引和切片
#类比于Python的列表，张量也支持索引和切片操作

#2.1.5 节省内存的操作

before = id(Y)
Y=X+Y
print(id(Y)==before)
#False，说明Y的内存地址发生了变化，说明Y被重新分配了内存空间
#类似cpp的深拷贝，重新分配内存空间。

Z=torch.zeros_like(Y)
print(id(Z))
Z[:] = X + Y
#X+=Y
print(id(Z))
#True，说明Z的内存地址没有发生改变


#2.1.6转换为其他Python对象
A=X.numpy()
B=torch.tensor(A)
print("type(A)",type(A))
print("type(B)",type(B))
#将大小为1的张量转换为Python标量
a=torch.tensor([3.5])
#a,a.item(),float(a),int(a)

