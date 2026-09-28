import numpy as np
import torch
import os
import pandas as pd

#2.2.1 读取数据集

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# 这行代码的作用是创建目录：
# os.path.join('..', 'data')：拼接路径，'..' 表示上一级目录，所以完整路径是 ../data（即当前文件的父目录下的 data 文件夹）
# os.makedirs()：创建目录（可以创建多级目录）
# exist_ok=True：如果目录已经存在，不会报错，直接忽略
# 简单来说：确保 ../data 这个文件夹存在，如果不存在就创建它。

data_file = os.path.join('..', 'data', 'house_tiny.csv')
# 这行代码的作用是构建文件路径：
# 将 '..'、'data'、'house_tiny.csv' 拼接成完整路径
# 结果是 ../data/house_tiny.csv
# 将这个路径赋值给变量 data_file，后续代码会用这个路径来读写数据文件

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

#2.2.2 处理缺失值
#插值法与均值填充法
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

#独热编码
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

#2.2.3 转换为张量格式
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)



