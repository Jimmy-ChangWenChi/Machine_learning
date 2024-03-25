# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:40:12 2024

@author: jimmy
"""

#Numpy產生樣本點

#產生 線性 樣本點
import numpy as np
# =============================================================================
# 
# sample = np.arange(0.,5.,0.2) # arange(下限,上限,間隔)
# #arange = Array Range
# print(sample)
# 
# #線性樣本點 + 洗牌
# np.random.shuffle(sample)
# print(sample)
# 
# #線性樣本點 + 更改維度
# sample = np.arange(0.0,5.0,0.2 )
# sample = sample.reshape(5,5)
# print(sample)
# 
# sample = sample.astype("unicode")
# print(sample)
# 
# =============================================================================


#產生 線性等間隔 樣本點
# =============================================================================
# sample1 = np.linspace(0.,5.,5)  #linspace(上限,下限,樣本數)
# print(sample1)
# 
# #產生 指數間隔 樣本點
# sample1 = np.logspace(0.,5.,5)  #logspace(上限,下限,樣本數)
# #結果 [10的零次方, 10的1.25次方, 10.的2.5次方,...]
# print(sample1)
# =============================================================================


# =============================================================================
# #產生 整數亂數 樣本點
# sample2 = np.random.randint(1,7,size=15) #產生[1,7) 之間的15個樣本數
# print(sample2)
# 
# #產生 浮點數數亂數 樣本點
# sample3 = np.random.rand(2,3) #產生[0,1)之間的浮點數 2*3個, 只需要維度
# print(sample3)
# 
# sample4 = np.random.rand(5)*3 #產生[0,3)之間的浮點數 1*5個
# print(sample4)
# 
# sample5 = np.random.rand(5)*3+2 #產生[2,5)之間的浮點數 1*5個
# print(sample5)
# =============================================================================

#亂數產生 類別資料 樣本點
# =============================================================================
# weather = ["Sunny","Cloudy","Rainy","Windy"]
# Taipei = np.random.choice(weather,size=(4,7),replace = True, p =[0.2,0.5,0.2,0.1]) #(類別資料陣列,陣列大小,True為重複挑選,機率)
# print(Taipei)
# =============================================================================

#產生 常態分布 樣本點

#標準常態分布點 平均值為0, 標準差為1
# =============================================================================
# normal1 = np.random.randn(3,5)
# print(normal1)
# 
# #一般常態分布 平均值不為0, 標準差不為1
# normal2 = np.random.normal(10,2,(3,5)) #平均值為10, 標準差為2
# print(normal2)
# =============================================================================

#取出 不重複 樣本點
dice1 = np.random.randint(1,7,15)
print(dice1)
unique1 = np.unique(dice1) #找出不重複樣本數
print(unique1)

unique1, const1 = np.unique(dice1,return_counts=True) #return_counts=True 不重複樣本數 出現次數
print(unique1,const1)
print(dict(zip(unique1,const1))) #合成字典











