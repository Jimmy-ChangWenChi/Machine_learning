# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 04:55:13 2024

@author: z7744
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:45:42 2024

@author: z7744
"""

# In[] Pre-processing
#from HappyML import preprocessor as pp
import HappyML.preprocessor as pp
# In[]讓使用者輸入下列資料：
# ser_gender = eval(input("請輸入您的性別（1.男  2.女）：")) - 1
# user_age = eval(input("請輸入您的年齡（6-15）："))
# user_height = eval(input("請輸入您的身高（cm）："))
# user_weight = eval(input("請輸入您的體重（kg）："))
user_gender=0
user_age=9
user_height=170
user_weight=55

# Dataset Loading
ds_Student_Height = pp.dataset("Student_Height.csv")
ds_Student_Weight = pp.dataset("Student_Weight.csv")

# Independent/Dependent Variables Decomposition
#X, Y = pp.decomposition(dataset, [0], [1])
SH_Years, SH_All_Average = pp.decomposition(ds_Student_Height, [1],[2])
SH_Man, SH_Woman = pp.decomposition(ds_Student_Height, [3],[4])
SW_Years, SW_All_Average = pp.decomposition(ds_Student_Weight, [1],[2])
SW_Man, SW_Woman = pp.decomposition(ds_Student_Weight, [3],[4])


# Split Training vs. Testing Set
#X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=2/3)
SH_Years_train, SH_Years_test, SH_Man_train, SH_Man_test = pp.split_train_test(SH_Years, SH_Man, train_size=0.8)


# Feature Scaling (optional), 特徵縮放(如果要畫圖必須要做特徵縮放, 怕每個資料比例尺差異太大, 劃出的圖會不好看)
# X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
# Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

# In[] Fitting Simple Regressor
# from sklearn.linear_model import LinearRegression

# regressor = LinearRegression()
# regressor.fit(X_train, Y_train) #訓練好的樣本
# Y_pred = regressor.predict(X_test) #未訓練樣本, pred是預測

# # 決定係數(coefficient of Determination):R2
# # 評估簡單線性模型結果, 殘差值越小越好, (每個真實值-平均值)平方, 等於0給100分(代表預測值接近真實值)
# # 1 - 0 = 1 = 100%
# R_Score = regressor.score(X_test, Y_test)
# print(R_Score)

# In[] Fitting Simple Regressor with HappyML
from HappyML.regression import SimpleRegressor

regressor = SimpleRegressor()
# 某些函數可以接火車, 意思訓練(X_train/Y_train訓練這個模型)+再丟預測值
Y_pred = regressor.fit(SH_Years_train, SH_Man_train).predict(SH_Years_test)
# 系統評分預測結果分數
print("R-Squared Score:", regressor.r_score(SH_Years_test, SH_Man_test))


# In[] Visualize the Model
import HappyML.model_drawer as md

#sample_data -> 樣本點
# md.sample_model(sample_data=(SH_Years_test, SH_Man_test), model_data=(SH_Years_test, Y_pred),
#                 title="測試集樣本點 vs. 預測模型", font="DFKai-sb") #測試結果

#sample_data(年齡(X軸),身高(Y軸))
md.sample_model(sample_data=(user_age, user_height), model_data=(SH_Years_test, Y_pred),
                title="身高落點模型", font="DFKai-sb") #測試結果