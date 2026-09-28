import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1. 一维数组
data=[1,2,3,4,5]
arr1=np.array(data)

#2. 二维数组
data2=[[1,2,3],
       [4,5,6],
       [5,6,7]]
arr2=np.array(data2)
print(arr2.shape)   #查看数组的形状
print(arr2.ndim)    #查看数组的维度

#3.特殊数组
x1=np.linspace(0,10,11)    #从0到10，生成10个均匀分布的点
x2=np.arange(0,10,1)       #从0到10，生成10个等差数列的点
data2=np.random.normal(loc=100,scale=10,size=200)        #正态分布均值100，标准差10，200个数据
plt.hist(data2,bins=20)  #频次分布直方图
plt.show()

#均匀分布:0到1之间的随机数
rand_data=np.random.rand(100)

#全0数组
zeros_array=np.zeros((3,4))
#全1数组
ones_array=np.ones((3,4))
#括号内写的是数组的形状


#4.数组的索引和切片
arr3=np.array([1,2,3,4])
print(arr3[0])  #索引为0的元素
print(arr3[1:3])  #索引为1到2的元素
print(arr3[1:])  #索引为1到3的元素
print(arr3[:3])  #索引为0到2的元素
print(arr3[::2])  #索引为偶数的元素
print(arr3[::-1])  #索引为倒序的元素

#5.条件筛选
arr3=np.array([1,2,3,4,5])
print(arr3[arr3>3])  #筛选出大于3的元素
print(arr3[arr3<=3])  #筛选出小于等于3的元素

#6.数组的操作
#形状转换(reshape)
a=np.array([1,2,3,4,5])
print(a.reshape(5,1))  #将数组转换为5行1列的矩阵
#转置
print(a.T)  #将数组的行和列交换
#合并
a1=np.array([1,2,3])
b1=np.array([4,5,6])
c1=np.concatenate([a1,b1])  #将两个数组合并
c2=np.vstack([a1,b1])  #将两个数组垂直合并

#7.数组的运算(数据批处理)
a1=np.array([1,2,3])
b1=np.array([4,5,6])
#对应元素处理
print(a1+b1)  #数组的加法
print(a1-b1)  #数组的减法
print(a1*b1)  #数组的乘法
print(a1/b1)  #数组的除法
print(a1**b1)  #数组的指数
#常见的数学函数
print(np.sin(a1))  #数组的正弦函数  
print(np.sqrt(a1))  #开方
print(np.exp(a1))  #数组的指数函数
print(np.log(a1))  #数组的对数函数
print(np.log2(a1))  #数组的二进制对数函数
print(np.log10(a1))  #数组的十进制对数函数 
#常见的统计函数
print(np.mean(a1))  #数组的平均值
print(np.median(a1))  #数组的中位数
print(np.std(a1))  #数组的标准差
print(np.var(a1))  #数组的方差
print(np.sum(a1))  #数组的和
print(np.prod(a1))  #数组的积
print(np.max(a1))  #数组的最大值
print(np.min(a1))  #数组的最小值
#二维数组
table=np.array([[1,2,3],
       [4,5,6],
       [5,6,7]])
print(np.mean(table,axis=0))  #按列求平均值
print(np.mean(table,axis=1))  #按行求平均值
















