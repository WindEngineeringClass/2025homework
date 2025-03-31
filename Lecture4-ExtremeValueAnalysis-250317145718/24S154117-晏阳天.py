#!/usr/bin/env python
# coding: utf-8

# # Question1
# Re-analyse the annual maximum gust wind speeds for (I) the years 1952 to 1998, (II) the years 1952 to 1997, i.e. ignore the high value recorded in 1998. Compare the resulting predictions of design wind speeds for (a) 50 years return period, and (b) 1000 years return period, and comment.
# # Question2
# Using the parameter estimation approach in code GB 50009 2012, predict the 10, 20, 50, 100, 200, 500, 1000 years return period design wind speeds for the above two cases, and compare the results with the Gumbel approach.
#

#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import pandas as pd
from io import StringIO
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 加载数据
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

# 将数据转换为DataFrame
df = pd.read_csv(StringIO(data))
df = df.sort_values(by="Wind speed m/s").reset_index(drop=True)
df_1952_1997 = df[df.year < 1998]  # 忽略1998年的高风速记录

# 定义重现期
return_periods = [10, 20, 50, 100, 200, 500, 1000]

# 定义Gumbel和Gringorten的缩减变量计算函数
def reduced_variate(data, method="Gumbel"):
    N = len(data)
    m = np.arange(1, N + 1)
    if method == "Gumbel":
        p = m / (N + 1)  # Gumbel方法
    elif method == "Gringorten":
        p = (m - 0.44) / (N + 0.12)  # Gringorten方法
    return -np.log(-np.log(p))

# 定义设计风速计算函数
def design_wind_speed(return_period, slope, intercept):
    return intercept + slope * (-np.log(-np.log(1 - 1 / return_period)))

# 定义GB 50009-2012设计风速计算函数
def GB_design_wind_speed(data, return_periods, N):
    mean = data['Wind speed m/s'].mean()
    std = data['Wind speed m/s'].std()
    coefficients = {47: (1.155374, 0.547192), 46: (1.153612, 0.546746)}
    if N not in coefficients:
        raise ValueError("N值无效，应为46或47")
    c1, c2 = coefficients[N]
    alpha = c1 / std
    u = mean - c2 / alpha
    results = [u + (-np.log(-np.log(1 - 1 / period))) / alpha for period in return_periods]
    return results

# 定义绘图函数
def plot_wind_speed(reduced_variate, wind_speeds, slope, intercept, method, years):
    plt.scatter(reduced_variate, wind_speeds, label=years, zorder=5)
    plt.plot(reduced_variate, slope * reduced_variate + intercept, '--', label=f'Fit line ({years})')
    plt.title(f'Gust Speed vs. Reduced Variate ({method})')
    plt.xlabel('Reduced Variate')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.xlim(-2, 5)
    plt.ylim(-10, 50)
    plt.show()

# 计算并绘制Gumbel和Gringorten结果
def analyze_data(data, years_label, method="Gumbel"):
    reduced_var = reduced_variate(data['Wind speed m/s'], method)
    slope, intercept, _, _, _ = linregress(reduced_var, data['Wind speed m/s'])
    plot_wind_speed(reduced_var, data['Wind speed m/s'], slope, intercept, method, years_label)
    return slope, intercept

# 分析1952-1998年和1952-1997年数据
print("Gumbel方法分析：")
gumbel_slope_1998, gumbel_intercept_1998 = analyze_data(df, "1952-1998", "Gumbel")
gumbel_slope_1997, gumbel_intercept_1997 = analyze_data(df_1952_1997, "1952-1997", "Gumbel")

print("Gringorten方法分析：")
gringorten_slope_1998, gringorten_intercept_1998 = analyze_data(df, "1952-1998", "Gringorten")
gringorten_slope_1997, gringorten_intercept_1997 = analyze_data(df_1952_1997, "1952-1997", "Gringorten")

# # Question1
def print_design_speeds(method, slope_1998, intercept_1998, slope_1997, intercept_1997):
    print(f"\n{method}方法设计风速：")
    for period in [50, 1000]:
        speed_1998 = design_wind_speed(period, slope_1998, intercept_1998)
        speed_1997 = design_wind_speed(period, slope_1997, intercept_1997)
        print(f"1952-1998年 {period}年重现期设计风速: {speed_1998:.2f} m/s")
        print(f"1952-1997年 {period}年重现期设计风速: {speed_1997:.2f} m/s")

print("问题1：比较50年和1000年重现期的设计风速")
print_design_speeds("Gumbel", gumbel_slope_1998, gumbel_intercept_1998, gumbel_slope_1997, gumbel_intercept_1997)
print_design_speeds("Gringorten", gringorten_slope_1998, gringorten_intercept_1998, gringorten_slope_1997, gringorten_intercept_1997)

# # Question2
print("\n问题2：使用GB 50009-2012方法预测设计风速")
print("1952-1998年：")
GB_results_1998 = GB_design_wind_speed(df, return_periods, 47)
print("\n1952-1997年：")
GB_results_1997 = GB_design_wind_speed(df_1952_1997, return_periods, 46)

# 计算Gumbel方法的设计风速
print("\nGumbel方法设计风速：")
print("1952-1998年：")
for period in return_periods:
    speed = design_wind_speed(period, gumbel_slope_1998, gumbel_intercept_1998)
    print(f"\t{period}年重现期设计风速: {speed:.2f} m/s")

print("\n1952-1997年：")
for period in return_periods:
    speed = design_wind_speed(period, gumbel_slope_1997, gumbel_intercept_1997)
    print(f"\t{period}年重现期设计风速: {speed:.2f} m/s")