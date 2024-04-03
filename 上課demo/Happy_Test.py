# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:54:30 2024

@author: jimmy
"""

from HappyML import preprocessor as pp

#資料前處理
dataset = pp.dataset("Salary_Data.csv")

X, Y = pp.decomposition(dataset, [0], [1])


# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)


#feature scaling 畫圖需要特徵縮放
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test)) #套用 X_train跟 X_test
Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

#用標準版做 簡單線性回歸
# =============================================================================
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train,Y_train) 
# Y_pred = regressor.predict(X_test) #模型預測答案
# 
# #殘差平方和
# R_Score = regressor.score(X_test,Y_test) #X_test 為預測值, Y_test 為真實值
# print(R_Score)
# =============================================================================

#用快樂版做 簡單線性回歸
from HappyML.regression import SimpleRegressor
regressor = SimpleRegressor()
Y_pred = regressor.fit(X_train,Y_train).predict(X_test)
print(regressor.r_score(X_test, Y_test)) #印出評估模型


#結果視覺化
from HappyML import model_drawer as md
sample_data=(X_train,Y_train) #樣本點
model_data = (X_train,regressor.predict(X_train)) #模型

md.sample_model(sample_data=sample_data, model_data=model_data, title="訓練集樣本點 VS 模型", font="DFKai-sb") #訓練結果
#sample 是樣本點

#md.sample_model(sample_data=(X_test,Y_test), model_data=(X_test,Y_pred), title="測試集樣本點 VS 模型", font="DFKai-sb") #測試結果

#md.sample_model(sample_data=(X_test,Y_test), model_data=(model_data), title="測試集樣本點 VS 模型1", font="DFKai-sb") #測試結果




