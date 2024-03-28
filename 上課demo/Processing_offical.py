#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:17:15 2024

@author: Jimmy
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("CarEvaluation.csv") #dataframe, 有欄位名稱

X = dataset.iloc[:, :-1].values #.values 可以將DataFrame中的值取出,變成NDArray, 機器學習只能接收NDArray, NDArray 沒有欄位名稱
Y = dataset.iloc[:, 4].values

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") 
#impute = 推算
#missing_values = np.nan 所有標示為 NaN的欄位
#strategy = mean, median, most_frequent, constant

imputer = imputer.fit(X[:,1:4])
#fit 根據陣列切片去算出結果

X[:,1:4] = imputer.transform(X[:,1:4])
#transform 將缺失資料,轉化為各欄平均值

#(類別)資料數位化
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(Y).astype("float64") # 將文字計算轉成數字,並用 64bit 浮點數顯示

ary_dummies = pd.get_dummies(X[:,0]).values
X = np.concatenate((ary_dummies,X[:,1:4]),axis = 1).astype("float64")

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#test_size = 0.2 等於 train_size = 0.8
#random_state 亂數產生器的亂數種子, 有助於每次切分 訓練集, 測試集的方法一樣


#特徵縮放
sc_X = StandardScaler().fit(X_train)
#StandardScaler物件產生 縮放計算 模型
X_train = sc_X.transform(X_train)
#縮放後的值, 替換X_train內的值
X_test = sc_X.transform(X_test)

