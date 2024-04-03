# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:22:30 2024

@author: jimmy
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#標準版寫法
#載入資料
dataset = pd.read_csv("HealthCheck.csv")
#print(dataset)

#切分自變數,應變數
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
#print(X)
#print(Y)

#缺失資料補缺
imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
#print(X)

#類別資料數位化
ary_dummies = pd.get_dummies(X[:,0]).values
X = np.concatenate((ary_dummies,X[:,1:3]),axis = 1).astype("float64")
#print(X)
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(Y).astype("float64")

#切分 訓練集,測試集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0) # 此為array, 並非DataFrame

#特徵縮放
sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

#print(type(X_train))
# print("自變數訓練集: " + X_train) #無法用 +號
# print("應變數訓練集: " + Y_train)
# print("自變數測試集: " + X_test)
# print("應變數測試集: " + Y_test)

print("自變數訓練集: ", X_train)
print("應變數訓練集: ", Y_train.reshape(1,-1))
print("自變數測試集: ", X_test)
print("應變數測試集: ", Y_test.reshape(1,-1))



#快樂版寫法
# =============================================================================
# import HappyML.preprocessor as pp
#  
# dataset = pp.dataset("HealthCheck.csv")
#  
# X, Y = pp.decomposition(dataset, x_columns = [i for i in range(3)], y_columns = [3])
#   #print(X)
#   #print(Y)
#  
# X = pp.missing_data(X, strategy="mean")
#  
# Y, Y_mapping = pp.label_encoder(Y, mapping=True)
#  
# X = pp.onehot_encoder(X, columns=[0])
#  
# X_train,X_test,Y_train,Y_test = pp.split_train_test(X,Y, train_size=0.8, random_state=0)
#  
# X_train,X_test = pp.feature_scaling(X_train, transform_arys=(X_train,X_test))
#  
# 
# print("自變數訓練集: ", X_train) 
# print("應變數訓練集: ", Y_train.T) #DataFrame 的行轉列, 需使用.T , reshape 是一般陣列才能用的
# print("自變數測試集: ", X_test)
# print("應變數測試集: ", Y_test.T)
# =============================================================================
