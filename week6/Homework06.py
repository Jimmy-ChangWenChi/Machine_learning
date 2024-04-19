# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 07:11:08 2024

@author: 黃文輝
"""

import pandas as pd
# # In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Device_Failure.csv")

# Decomposition of Variables
X, Y = pp.decomposition(dataset, x_columns=[0], y_columns=[1])

# Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

# In[] Polynomial Regression with HappyML's Class 系統找出最佳解
from HappyML.regression import PolynomialRegressor
import HappyML.model_drawer as md
from HappyML.performance import rmse

model = PolynomialRegressor()
# verbose=ture 顯示計算過程
# degree 一般不超過10
model.best_degree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose=True)
Y_pred = model.fit(x_train=X, y_train=Y).predict(x_test=X)
md.sample_model(sample_data=(X, Y), model_data=(X, Y_pred), xlabel=f"Degree={model.degree}  RMSE={rmse(Y, Y_pred):.4f}")

# RMSE是殘差, N+1, 一下次RMSE往上升, 代表RMSE最佳解是N-1
# 如果RMSE, 跟上一次RMSE差異太大，就不能採用那個Degree
print(f"Degree={model.degree}  RMSE={rmse(Y, Y_pred):.4f}")

# In[] 取得二維數字型態資料
# 老師提示的寫法
# [[User_Input_Year]]
# pd.DataFrame([[User_Input_Year]])
# model_User_Input_Year.predict(pd.DataFrame([[User_Input_Year]])).iloc[0,0] #取得總失效時間
# In[] 預設使用者輸入值
# User_Input_Year=13
# 讓使用者輸入資料：
User_Input_Year = eval(input("請輸入設備已使用年份:"))

# In[] Mark test program
# model_User_Input_Year = PolynomialRegressor()
# model_User_Input_Year.best_degree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose=False)
# Y_User_Input_Year_pred = model_User_Input_Year.fit(x_train=X, y_train=Y).predict(x_test=pd.DataFrame([[User_Input_Year]]))
# print("您的設備預測總失效時間 = ", "{:.4f}".format(Y_User_Input_Year_pred.iloc[0,0]), "小時")
# print("平均每年失效時間 = ", "{:.4f}".format(Y_User_Input_Year_pred.iloc[0,0]/User_Input_Year), "小時/年")

# In[] got user input 預測總失效時間 and 平均每年失效時間 
[[User_Input_Year]]
pd.DataFrame([[User_Input_Year]])
print("您的設備預測總失效時間 = ", "{:.4f}".format(model.predict(pd.DataFrame([[User_Input_Year]])).iloc[0,0]), "小時")
print("平均每年失效時間 = ", "{:.4f}".format(model.predict(pd.DataFrame([[User_Input_Year]])).iloc[0,0]/User_Input_Year), "小時/年")











