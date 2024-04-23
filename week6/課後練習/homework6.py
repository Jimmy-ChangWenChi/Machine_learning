# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:32:50 2024

@author: jimmy
"""

import HappyML.preprocessor as pp
import pandas as pd
from HappyML.performance import rmse

dataset = pp.dataset("Device_Failure.csv")


X, Y = pp.decomposition(dataset, [0],[1])

X_train, X_test,Y_train,Y_test = pp.split_train_test(X, Y, train_size=0.8)


from HappyML.regression import PolynomialRegressor
import HappyML.model_drawer as md

model = PolynomialRegressor()
model.best_degree(X_train, Y_train, X_test, Y_test,verbose=True)

Y_pred = model.fit(X, Y).predict(X)

md.sample_model(sample_data=(X,Y), model_data=(X,Y_pred),xlabel=f"The best degree is {model.degree}, RMSE={rmse(Y, Y_pred):.4f}")


print(f"The best degree is {model.degree}")
print(f"RMSE={rmse(Y, Y_pred):.4f}") #陣列大小需依樣


input_year = eval(input("目前使用年份: "))

#[[input_year]] #轉成二維陣列
pd_year = pd.DataFrame([[input_year]]) #轉成1乘1


print(f"你的設備預測總失效時間 = {model.predict(pd_year).iloc[0,0]} 小時")
#print(f"你的設備預測總失效時間 = ", "{:.4f}".format(model.fit(X,Y).predict(pd_year).iloc[0,0]),"小時" )

print(f"平均每年失效時間 = ", "{:.4f}".format(model.predict(pd_year).iloc[0,0]/input_year),"小時/年" )