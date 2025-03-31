#!/usr/bin/env python
# coding: utf-8

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt  
from scipy.stats import linregress  
from io import StringIO

# 读取数据
data = """
year,Wind speed m/s
1952,31.4
1953,33.4
1954,29.8
1955,30.3
1956,27.8
1957,30.3
1958,29.3
1959,36.5
1960,29.3
1961,27.3
1962,31.9
1963,28.8
1964,25.2
1965,27.3
1966,23.7
1967,27.8
1968,32.4
1969,27.8
1970,26.2
1971,30.9
1972,31.9
1973,27.3
1974,25.7
1975,32.9
1976,28.3
1977,27.3
1978,28.3
1979,28.3
1980,29.3
1981,27.8
1982,27.8
1983,30.9
1984,26.7
1985,30.3
1986,28.3
1987,30.3
1988,34
1989,28.8
1990,30.3
1991,27.3
1992,27.8
1993,28.8
1994,30.9
1995,26.2
1996,25.7
1997,24.7
1998,42.2
"""
df = pd.read_csv(StringIO(data))

# 计算 Reduced Variate
def compute_reduced_variate(data, method="Gumbel"):
    N = len(data)
    m = np.arange(1, N + 1)
    if method == "Gumbel":
        p = m / (N + 1)
    elif method == "Gringorten":
        p = (m - 0.44) / (N + 0.12)
    return -np.log(-np.log(p))

# 计算并绘制回归线
def plot_gumbel_fit(df, title, color, linestyle):
    reduced_variate = compute_reduced_variate(df["Wind speed m/s"])
    wind_speeds = df["Wind speed m/s"].values
    slope, intercept, _, _, _ = linregress(reduced_variate, wind_speeds)
    
    plt.scatter(reduced_variate, wind_speeds, color=color, label=f'{title}', zorder=5)
    plt.plot(reduced_variate, slope * reduced_variate + intercept, linestyle, label=f'Fit line ({title})')

    return slope, intercept

# 计算回归
df_1952_1997 = df[df.year < 1998]
plt.figure(figsize=(12, 5))

# 1952-1998 数据
slope_1998, intercept_1998 = plot_gumbel_fit(df, "1952-1998", "orange", "r--")

# 1952-1997 数据（忽略1998）
slope_1997, intercept_1997 = plot_gumbel_fit(df_1952_1997, "1952-1997", "blue", "g--")

plt.title('Gust Speed vs. Reduced Variate (Gumbel)')
plt.xlabel('Reduced Variate')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.grid()
plt.show()

# 计算返回周期设计风速
def design_wind_speed(return_period, slope, intercept):
    y = -np.log(-np.log(1 - 1 / return_period))
    return intercept + slope * y

# 计算 50 年 & 1000 年返回期风速
for period in [50, 1000]:
    speed_1998 = design_wind_speed(period, slope_1998, intercept_1998)
    speed_1997 = design_wind_speed(period, slope_1997, intercept_1997)
    print(f"{period}年返回期风速: 1952-1998 = {speed_1998:.2f} m/s, 1952-1997 = {speed_1997:.2f} m/s")

# GB 50009 2012 方法计算风速
def GB_design_wind_speed(data, return_periods):
    N = len(data)
    mean = data['Wind speed m/s'].mean()
    std = data['Wind speed m/s'].std()

    # 根据 N 自动获取系数
    coefficients = {47: (1.155374, 0.547192), 46: (1.153612, 0.546746)}
    if N not in coefficients:
        raise ValueError("N 超出支持范围，应为 46 或 47")

    c1, c2 = coefficients[N]
    alpha = c1 / std
    u = mean - c2 / alpha

    # 计算各个返回期的设计风速
    return {T: u + (-np.log(-np.log(1 - 1 / T))) / alpha for T in return_periods}

return_periods = [10, 20, 50, 100, 200, 500, 1000]
gb_speeds_1998 = GB_design_wind_speed(df, return_periods)
gb_speeds_1997 = GB_design_wind_speed(df_1952_1997, return_periods)

# 显示计算结果
print("\nGB 50009 2012 计算结果（m/s）")
for T in return_periods:
    print(f"返回期 {T} 年: 1952-1998 = {gb_speeds_1998[T]:.2f}, 1952-1997 = {gb_speeds_1997[T]:.2f}")
