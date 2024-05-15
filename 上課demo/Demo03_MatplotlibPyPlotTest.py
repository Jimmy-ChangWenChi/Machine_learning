# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:55:17 2019

@author: 俊男
"""

# In[1]
import matplotlib.pyplot as plt
import numpy as np

# In[] Line Chart

# 1D
plt.plot([2, 4, 6, 8])
plt.show()

# 2D
plt.plot([5, 10, 15, 20], [2, 4, 6, 8])
plt.show()

# 2D with format
plt.plot([5, 10, 15, 20], [2, 4, 6, 8], "ro")
plt.show()

X = np.arange(0., 5., 0.2)
plt.plot(X, X, "r--")
plt.plot(X, X**2, "bs")
plt.plot(X, X**3, "g^")
plt.show()

# In[] Chinese Labeling & Legends

plt.rcParams['font.sans-serif']=['DFKai-sb']
plt.rcParams['axes.unicode_minus'] = False

plt.title("X, X平方, X三次方線圖")
plt.xlabel("X 軸")
plt.ylabel("Y 軸")

plt.plot(X, X, "r--", label="X")
plt.plot(X, X**2, "bs", label="X平方")
plt.plot(X, X**3, "g^", label="X立方")
plt.legend(loc='best')

plt.show()

# In[] Bar Chart

X = ["蘋果", "三星", "OPPO", "華為", "華碩", "小米", "SONY", "HTC", "Sugar", "Nokia"]
Y = [24.1, 23.3, 9.2, 8.9, 8.1, 6.7, 6.5, 4.6, 1.7, 1.0]

plt.title("2019年4月 台灣地區手機佔有率")
plt.xlabel("手機品牌")
plt.ylabel("百分比")

plt.bar(X, Y)
plt.show()

# In[] Scatter Chart

# Size of Sample Data
n = 150

# Group1 Data around (-1, 2) as Normal Distribution
X1 = np.random.normal(-1, 1, n)
Y1 = np.random.normal(2, 1, n)

# Group2 Data around (2, -1) as Normal Distribution
X2 = np.random.normal(2, 1, n)
Y2 = np.random.normal(-1, 1, n)

# Scatter Chart with Size of marker (s), Color of marker (c), and Style of marker (marker)
plt.scatter(X1, Y1, s=75, c="red", marker="+")
plt.scatter(X2, Y2, s=75, c="blue", marker="o")
plt.show()
