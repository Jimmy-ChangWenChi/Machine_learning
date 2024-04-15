# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:12:40 2024

@author: 黃文輝
"""

import pandas as pd
# In[] Pre-processing
#from HappyML import preprocessor as pp
import HappyML.preprocessor as pp

# In[]讓使用者輸入下列資料：
# ser_gender = eval(input("請輸入您的性別（1.男  2.女）：")) - 1
# user_age = eval(input("請輸入您的年齡（6-15）："))
# user_height = eval(input("請輸入您的身高（cm）："))
# user_weight = eval(input("請輸入您的體重（kg）："))
user_gender=0
user_age=15
user_height=170
user_weight=55

# Dataset Loading
ds_Student_Height = pp.dataset("Student_Height.csv")
ds_Student_Weight = pp.dataset("Student_Weight.csv")

# In[] Independent/Dependent Variables Decomposition
SH_Years, SH_All_Average = pp.decomposition(ds_Student_Height, [1],[2])
SH_Man, SH_Woman = pp.decomposition(ds_Student_Height, [3],[4])
SW_Years, SW_All_Average = pp.decomposition(ds_Student_Weight, [1],[2])
SW_Man, SW_Woman = pp.decomposition(ds_Student_Weight, [3],[4])


# In[] Split Training vs. Testing Set
SH_Years_train, SH_Years_test, SH_Man_train, SH_Man_test = pp.split_train_test(SH_Years, SH_Man, train_size=0.8)


# In[] Fitting Simple Regressor with HappyML
from HappyML.regression import SimpleRegressor

# 訓練出四個「簡單線性迴歸器」：
# regressor = [[SimpleRegressor(), SimpleRegressor()], [SimpleRegressor(), SimpleRegressor()]]
# regressor[0][0] -->年齡 vs. 男生身高
# regressor[0][1] -->年齡 vs. 女生身高
# regressor[1][0] -->年齡 vs. 男生體重
# regressor[1][1] -->年齡 vs. 女生體重

#simple_reg.fit(X_train, Y_train).predict(X_test)
regressor = [[SimpleRegressor(), SimpleRegressor()], [SimpleRegressor(), SimpleRegressor()]]




regressor[0][user_gender]=regressor[0][user_gender].fit(SH_Years_train, SH_Man_train).predict(SH_Years_test)


# In[] Visualize the Model
import HappyML.model_drawer as md
#sample_data -> 樣本點
# md.sample_model(sample_data=(SH_Years_test, SH_Man_test), model_data=(SH_Years_test, Y_pred),
#                 title="測試集樣本點 vs. 預測模型", font="DFKai-sb") #測試結果

#sample_data(年齡(X軸),身高(Y軸))
#print(regressor[0][user_gender])
md.sample_model(sample_data=(user_age, user_height), model_data=(SH_Years_test, regressor[0][user_gender]),
                title="身高落點模型", font="DFKai-sb") #測試結果

# #print(regressor[0][user_gender].iloc[0, 0])


# Y_pred = regressor.fit(X_train, Y_train).predict(X_test)
# print("R-Squared Score:", regressor.r_score(X_test, Y_test))


# In[] 老師提示
# regressor[0][user_gender].predict(pd.DataFrame([[user_age]]))
# pd.DataFrame([[user_age]])
# 可用下列方法，取得使用者同齡之平均身高、體重：
# h_avg = regressor[0][user_gender].predict(x_test=pd.DataFrame([[age]])).iloc[0, 0]
# w_avg = regressor[1][user_gender].predict(x_test=pd.DataFrame([[age]])).iloc[0, 0]

df_user_age=pd.DataFrame([[user_age]])


h_avg = regressor[0][user_gender].predict(SH_Years_test(df_user_age)).iloc[0, 0]
#w_avg = regressor[1][user_gender].predict(SH_Years_=pd.DataFrame([[user_age]])).iloc[0, 0]