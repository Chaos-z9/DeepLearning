import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate#插值模块
from scipy import signal#信号处理模块(去噪)
from scipy.ndimage import median_filter #中值滤波
from scipy.ndimage import gaussian_filter #高斯滤波
from scipy.optimize import curve_fit #曲线拟合
from scipy import integrate #积分模块


plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus']=False

#1.插值(插值的用法?)
time=np.array([0,1,2,3,4,5,6,7])
temp=np.array([20,22,24,26,28,23,21,19])
time_dense=np.linspace(0,7,100)
#1.1 线性插值
temp_inter1=interpolate.interp1d(time,temp,kind='linear')
#1.2 二次插值
temp_inter2=interpolate.interp1d(time,temp,kind='quadratic')
#1.3 三次插值
temp_inter3=interpolate.interp1d(time,temp,kind='cubic')
plt.figure(figsize=(10,6))
plt.scatter(time,temp,s=40,c='red',label='原始数据')
plt.plot(time_dense,temp_inter1(time_dense),label='线性插值数据')
plt.plot(time_dense,temp_inter2(time_dense),label='二次插值数据')
plt.plot(time_dense,temp_inter3(time_dense),label='三次插值数据')
plt.xlabel('时间')
plt.ylabel('温度')
plt.title('温度变化')
plt.legend()
plt.grid(True,alpha=0.5)
#plt.savefig('插值.png',dpi=300)

#2.去噪(去噪的用法?)
# 生成带噪声的数据
time = np.linspace(0, 20, 200)  # 24小时，200个数据点
# 真实温度变化（昼夜周期 + 随机波动）
true_temp = 10 * np.sin(2 * np.pi * time / 24)
# 添加测量噪声
noise = np.random.randn(200)  # 高斯噪声
# 观测数据 = 真实值 + 噪声
observed_temp = true_temp + noise

# 去噪
#Savitzky-Golay滤波器(原理是什么?)
window_length = 12  #窗口长度‘
polyorder = 2  #多项式阶数
savgol_smooth=signal.savgol_filter(observed_temp,window_length,polyorder)
# 中值滤波(原理是什么?)
kernel_size = 30  # 核大小
median_smooth = median_filter(observed_temp, kernel_size)
# 高斯滤波(原理是什么?)
sigma = 5  # 标准差
gaussian_smooth = gaussian_filter(observed_temp, sigma)

plt.figure(figsize=(8, 4))
plt.plot(time, true_temp, 'k-', linewidth=2, label='真实值')
plt.plot(time, observed_temp, 'b-', linewidth=1.5, label='添加噪声')
plt.plot(time, savgol_smooth, 'r-', linewidth=1.5, label='Savitzky-Golay滤波')
plt.plot(time, median_smooth, 'g-', linewidth=1.5, label='中值滤波')
plt.plot(time, gaussian_smooth, 'y-', linewidth=1.5, label='高斯滤波')

plt.xlabel('时间（小时）')
plt.ylabel('温度（℃）')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
#plt.savefig('去噪.png',dpi=300)
plt.show()

#3.曲线拟合(曲线拟合的用法?)
x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
y_data = np.array([10, 6.1, 3.7, 2.2, 1.4, 0.8, 0.5, 0.3])

#3.1 线性拟合
# 定义线性函数模型（y = ax + b）
def linear(x, a, b):
    return a * x + b

# 生成100个等间距的x值（0到7之间），用于绘制平滑的拟合曲线
# 参数说明：起点=0, 终点=7, 点的数量=100
x_fit = np.linspace(0, 7, 100)

# 使用curve_fit进行曲线拟合，自动寻找最佳的a和b参数
# 参数1 linear: 要拟合的函数模型
# 参数2 x_data: 实验数据的x值
# 参数3 y_data: 实验数据的y值
# 返回值1 params_liner: 拟合出的最佳参数数组 [a, b]
# 返回值2 _: 参数的协方差矩阵（衡量拟合质量，此处不需要）
params_liner, _ = curve_fit(linear, x_data, y_data)

# 解包拟合参数，将最佳参数赋值给变量
a_lin, b_lin = params_liner

# 使用拟合出的最佳参数a和b，计算100个密集点对应的y值
# 这些点将用于绘制平滑的拟合直线
pred_lin = linear(x_fit, a_lin, b_lin)

#3.2二次拟合
# 定义二次函数模型（y = ax^2 + bx + c）
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

params_quad, _ = curve_fit(quadratic, x_data, y_data)
a_quad, b_quad, c_quad = params_quad
pred_quad = quadratic(x_fit, a_quad, b_quad, c_quad)

#3.3指数拟合
# 定义指数函数模型（y = a * b^x）
def exponential(x, a, b):
    return a * b**x

params_exp, _ = curve_fit(exponential, x_data, y_data)
a_exp, b_exp = params_exp
pred_exp = exponential(x_fit, a_exp, b_exp)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, color='black', label='实验数据')
plt.plot(x_fit, pred_lin, color='red', label='线性拟合')
plt.plot(x_fit, pred_quad, color='blue', label='二次拟合')
plt.plot(x_fit, pred_exp, color='green', label='指数拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('曲线拟合')
#plt.savefig('曲线拟合.png',dpi=300)
plt.show()



#4.积分
x_data = np.linspace(0, 3, 30)
y_data = 2 + np.sin(x_data)
areas=integrate.trapezoid(y_data,x_data)
plt.figure()
# 绘制曲线
plt.plot(x_data, y_data, 'b-', linewidth=2, label='实验数据', marker='o', markersize=4)
plt.fill_between(x_data, y_data, color='lightblue', alpha=0.3)
plt.axhline(areas, color='r', linestyle='--', label='积分值')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.title(f'积分值{areas:.2f}')
#plt.savefig('积分.png',dpi=300)
plt.show()

#5.微分
# 生成数据
t = np.linspace(0, 10, 100)  # 时间 0‑10秒
x = 2*t**2 + 3*t + 5  # 位移：2t² + 3t + 5
x_noisy = x + np.random.randn(100)  # 加噪声

# 计算导数
dt = t[1] - t[0]  # 时间间隔
v = signal.savgol_filter(x_noisy, 21, 3, deriv=1, delta=dt)  # 速度
#v = signal.savgol_filter(x_noisy, 21(窗口大小), 3(多项式阶数), deriv=1(求1阶导数), delta=dt)
plt.figure(figsize=(6, 5))

# 第一个子图：位移
plt.subplot(2, 1, 1)
plt.plot(t, x, 'b-', label='理论')
plt.plot(t, x_noisy, 'r.-', alpha=0.6, label='测量')
plt.ylabel('位移 (m)')
plt.legend()
plt.grid(True, alpha=0.3)

# 第二个子图：速度
plt.subplot(2, 1, 2)
plt.plot(t, 4*t+3, 'b-', label='理论')
plt.plot(t, v, 'r.-', label='计算')
plt.ylabel('速度 (m/s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlabel('时间 (s)')
#plt.savefig('微分.png',dpi=300)
plt.show()





