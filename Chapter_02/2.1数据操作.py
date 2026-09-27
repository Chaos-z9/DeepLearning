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
Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((X,Y),dim=0))
print(torch.cat((X,Y),dim=1))


