import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus']=False

#1.读取数据
#读取txt/csv文件
# df=pd.read_csv(
#     "data.csv",#文件路径
#     sep="\t",   #分隔符
#     header=0,   #表头行
#     encoding="utf-8"
#     na_values=" "  #识别空格缺失值标记
# )
# print(df)

#读取excel文件
df=pd.read_excel(
    r"Data\25_gc.xlsx",#文件路径
    sheet_name="男胎检测数据",#表名
    header=0,   #表头行
    #encoding="utf-8"     这个函数不需要
)
print(df)
#2.查看数据
print(df["末次月经"])
print(df.iloc[:,6])
#Eg:查看第1行第7列的单个值
# df.iloc[0, 6]      # 第1行第7列的单个值
# df.iloc[0:5, 6]    # 前5行的第7列
# df.iloc[:, 0:3]    # 所有行的前3列
# df.iloc[2:5, :]    # 第3-5行的所有列



#3.查看数据信息
# 看数据形状: (行数, 列数), 确认有没有漏读
print(df.shape)
# 查看行索引
print(df.index)
# 查看所有列名（表头）
print(df.columns)
# 看前5行数据，确认列名、数据有没有读对
df.head()
# 看最后3行数据
df.tail(3)
# 查看每列的数据类型、缺失值数量
df.info()
# 自动计算所有数值列的均值、标准差、最大/最小值、中位数
df.describe()


# #4.数据选择与筛选
# # 选择单列(返回Series对象)
# NumPregnancies=df["怀孕次数"]
# print(NumPregnancies)
# # 选择多列(返回DataFrame对象)
# data_plot=df[["身高","体重"]]
# plt.figure()
# plt.scatter(data_plot["身高"],data_plot["体重"],s=50,color='b',marker='o',label='实验点')
# plt.legend(['实验点'])
# plt.title('身高体重关系')
# plt.xlabel('身高')
# plt.ylabel('体重')
# plt.xlabel('身高')
# plt.ylabel('体重')
# plt.show()



#5.数据清洗
# 缺失值处理
#查看每列的缺失值数量(先看情况，再决定怎么处理)
print(df.isnull().sum())

#方法1:删除缺失值
df_clear=df.dropna(inplace=True)

#方法2:填充缺失值
#用均值填充缺失值
#df["身高"]=df["身高"].fillna(df["身高"].mean(),inplace=True)
#用中位数填充缺失值
#df["身高"]=df["身高"].fillna(df["身高"].median(),inplace=True)

#重复值处理
#查看重复值
print(df.duplicated().sum())
#删除重复行
df_clear=df.drop_duplicates(inplace=True)



#6.数据计算与新增列
df['XXX']=df['身高']/(df['体重']*2)
#正常的计算
#df['BMI']=df['身高']**2/df['体重']
# print(df["应力(MPa)"].mean())   # 平均值
# print(df["应力(MPa)"].std())    # 标准差（论文常用）
# print(df["应力(MPa)"].max())    # 峰值应力（材料实验常用）
# print(df["应力(MPa)"].min())    # 最小值
# print(df["应力(MPa)"].sum())    # 求和
# print(df["应力(MPa)"].median()) # 中位数




