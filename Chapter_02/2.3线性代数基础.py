import torch

#2.3.1标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, x * y, x / y, x**y)
#一维张量

#2.3.2. 向量
x=torch.tensor([1,2])
print(x)
print(x[0])#下标访问


#2.3.2.1. 长度、维度和形状
print(x.shape)#向量的形状
print(len(x))#向量的长度

#2.3.3. 矩阵
A=torch.tensor([[1,2],[3,4]])
print(A)
print(A.shape)#矩阵的形状
print(A.T)#矩阵的转置
#对称矩阵的A==A.T


#2.3.4. 张量
#就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构。 张量（本小节中的“张量”指代数对象）是描述具有任意数量轴的
#维数组的通用方法。 例如，向量是一阶张量，矩阵是二阶张量。
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(X.shape)


#2.3.5. 张量算法的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B,A*B)
print("A的形状:", A.shape)
print("B的形状:", B.shape)
print("A + B的形状:", A + B)
print("A * B(Hadamard积)的形状:", A * B)
#将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘
print("A * 2(Hadamard积)的形状:", (A * 2).shape)
print("A + 2的形状:", (A + 2).shape)


#2.3.6. 降维
#求和降维
print("A的求和结果:", A.sum())
print("A的求和结果的形状:", A.sum().shape)
#对轴求和
print("A对轴0求和结果:", A.sum(dim=0))
print("A对轴0求和结果的形状:", A.sum(dim=0).shape)
print("A对轴1求和结果:", A.sum(dim=1))
print("A对轴1求和结果的形状:", A.sum(dim=1).shape)
#平均数求和
print("A对轴0求平均数结果:", A.mean(dim=0))


#2.3.6.1. 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(sum_A.shape)
print("利用广播机制:", A / sum_A)
#cumsum函数:此函数不会沿任何轴降低输入张量的维度。
print("A的轴0累计和结果:", A.cumsum(dim=0))
print("A的轴0累计和结果的形状:", A.cumsum(dim=0).shape)
print("A的轴1累计和结果:", A.cumsum(dim=1))
print("A的轴1累计和结果的形状:", A.cumsum(dim=1).shape)



#2.3.7. 点积（Dot Product）
#相同位置的按元素乘积的和
x=torch.tensor([1,2,3,4,5])
y=torch.tensor([0,1,1,1,1])
print("x",x)
print("y",y)
print("x和y的点积:",torch.dot(x,y))
print("x和y的点积的sum形式:",torch.sum(x * y))


#2.3.8. 矩阵-向量积
#矩阵A的第i行与向量x的点积，得到A的第i行的线性组合
print("A,x的矩阵-向量积",torch.mv(A,x))

#2.3.9. 矩阵-矩阵乘法
C=torch.arange(12).reshape(3,4)
D=torch.arange(8).reshape(4,2)
print("C",C)
print("D",D)
print("C,D的矩阵-矩阵乘法",torch.mm(C,D))


#2.3.10. 向量的范数（Norm）
u = torch.tensor([3.0, -4.0])
print("u的2范数向量:", torch.norm(u))




