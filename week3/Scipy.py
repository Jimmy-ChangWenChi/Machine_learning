# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:28:59 2024

@author: jimmy
"""
import numpy as np
from scipy.spatial.distance import pdist,squareform
#import scipy as sp

#scipy 子套件

# =============================================================================
# constants 各種物理與數學上的常數
# io 檔案讀寫工具
# sparse 稀疏矩陣工具
# interpolate 內差法套件
# stats 統計相關套件
# linalg 線性代數套件
# cluster 集群問題套件
# optimize 最佳解套件
# integrate 積分套件
# fftpack 快速傅立葉轉換套件
# signal 數位信號處理套件
# ndimage 多為影像處理套件
# =============================================================================

x = np.array([[0,1],[1,0],[2,0]])

x = sp.spatial.distance.pdist(x,'euclidean')
print(sp.spatial.distance.squareform(x))

x = pdist(x,'euclidean') # 各點之間幾何距離
#print(squareform(x))


print(x)














