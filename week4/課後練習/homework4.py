# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:41:06 2024

@author: jimmy
"""

import pandas as pd
import HappyML.preprocessor as pp
import warnings

warnings.filterwarnings("ignore")

user_Gender = eval(input("請輸入您的性別(1.男 2.女) : "))-1
user_age = eval(input("請輸入您的年齡(6-15) : "))
user_height = eval(input("請輸入您的身高(cm) : "))
user_weight = eval(input("請輸入您的體重(kg) : "))


#HEIGHT == 0,WEIGHT == 1,MALE == 0, FEMALE == 1

#資料前處理
dataset_Height = pp.dataset("Student_Height.csv")
dataset_Weight = pp.dataset("Student_Weight.csv")


# X, Y = pp.decomposition(dataset_Height, [1], [2])
# X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

All_Age, All_Height = pp.decomposition(dataset_Height,[1],[2])
Man_Height,Woman_Height = pp.decomposition(dataset_Height, [3],[4])

All_Age, All_Weight = pp.decomposition(dataset_Weight,[1],[2])
Man_Weight,Woman_Weight = pp.decomposition(dataset_Weight, [3],[4])

if user_Gender == 0:
    All_Age_train, All_Age_test, Man_Height_train, Man_Height_test = pp.split_train_test(All_Age, Man_Height, train_size=0.8)
    All_Age_train, All_Age_test, Man_Weight_train, Man_Weight_test = pp.split_train_test(All_Age, Man_Weight, train_size=0.8)
else:
    All_Age_train, All_Age_test, Woman_Height_train, Woman_Height_test = pp.split_train_test(All_Age, Woman_Height, train_size=0.8)
    All_Age_train, All_Age_test, Woman_Weight_train, Woman_Weight_test = pp.split_train_test(All_Age, Woman_Weight, train_size=0.8)




from HappyML.regression import SimpleRegressor


regressor = SimpleRegressor()

#身高
if user_Gender == 0:
    Y_pred_Height = regressor.fit(All_Age_train, Man_Height_train).predict(All_Age_test)
    # 預測結果分數
    print("R-Squared Score:", regressor.r_score(All_Age_test, Man_Height_test))

    #體重
    Y_pred_Weight = regressor.fit(All_Age_train, Man_Weight_train).predict(All_Age_test)
    print("R-Squared Score:", regressor.r_score(All_Age_test, Man_Weight_test))

else:
    Y_pred_Height = regressor.fit(All_Age_train, Woman_Height_train).predict(All_Age_test)
    # 預測結果分數
    print("R-Squared Score:", regressor.r_score(All_Age_test, Woman_Height_test))

    #體重
    Y_pred_Weight = regressor.fit(All_Age_train, Woman_Weight_train).predict(All_Age_test)
    print("R-Squared Score:", regressor.r_score(All_Age_test, Woman_Weight_test))



from HappyML import model_drawer as md
md.sample_model(sample_data=(user_age, user_height), model_data=(All_Age_test, Y_pred_Height),
                title="身高落點模型", font="DFKai-sb") #測試結果

md.sample_model(sample_data=(user_age, user_weight), model_data=(All_Age_test, Y_pred_Weight),
                title="體重落點模型", font="DFKai-sb") #測試結果

