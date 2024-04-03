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

X, Y = pp.decomposition(dataset_Height, [1], [3])
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test)) #套用 X_train跟 X_test
Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

from HappyML.regression import SimpleRegressor
# regressor =[[SimpleRegressor(),SimpleRegressor()],[SimpleRegressor(),SimpleRegressor()]]
# Y_pred = regressor[0][user_Gender].predict(pd.DataFrame([[user_Gender]])).iloc[0,0]

regressor = SimpleRegressor()
Y_pred = regressor.fit(X_train, Y_train).predict(X_test)


print(regressor.r_score(X_test, Y_test))

from HappyML import model_drawer as md
sample_data = (user_age,user_height) #樣本點

model_data = (X_train,regressor.predict(X_train)) #模型
md.sample_model(sample_data=sample_data, model_data=model_data, title="男生身高訓練集樣本點 VS 模型", font="DFKai-sb")




# X_AGE, Y_MALE_Height = pp.decomposition(dataset_Height, [1], [3])
# X_AGE, Y_FEMALE_Height = pp.decomposition(dataset_Height, [1], [4])
# X_AGE, Y_MALE_Weight = pp.decomposition(dataset_Weight, [1], [3])
# X_AGE, Y_FEMALE_Weight = pp.decomposition(dataset_Weight, [1], [4])

# =============================================================================
#X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)
# X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)
# X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)
# X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)
# =============================================================================

# #特徵縮放
# X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test)) #套用 X_train跟 X_test
# Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))


# from HappyML.regression import SimpleRegressor
# #regressor = SimpleRegressor()
# regressor =[[SimpleRegressor(),SimpleRegressor()],[SimpleRegressor(),SimpleRegressor()]]

# #Y_pred = regressor.fit(X_train,Y_train).predict(X_test)
# Y_pred = regressor[HEIGHT][user_Gender].predict(pd.DataFrame([[user_Gender]])) #第一組[] 是控制身高體重, 第二組[]是控制男生女生

# print(regressor.r_score(X_test, Y_test))

# from HappyML import model_drawer as md
# sample_data=(X_train,Y_train) #樣本點
# model_data = (X_train,regressor.predict(X_train)) #模型

# md.sample_model(sample_data=sample_data, model_data=model_data, title="訓練集樣本點 VS 模型", font="DFKai-sb") #訓練結果



# =============================================================================
# HEIGHT = 0,WEIGHT = 1,MALE = 0, FEMALE = 1
# 
# pd.DataFrame([[user_age]])
# 
# 
# regressor[HEIGHT][user_gender].predict(pd.DataFrame([[user_age]])).iloc[0,0]
# =============================================================================


