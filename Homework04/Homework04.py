# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:59:21 2024

@author: 黃文輝
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:12:40 2024

@author: 黃文輝
"""

import pandas as pd
# In[] Pre-processing
#from HappyML import preprocessor as pp
import HappyML.preprocessor as pp

# In[] 老師提示
# 讓使用者輸入下列資料：
# 預設使用者輸入值
# user_gender=1
# user_age=10
# user_height=138
# user_weight=45
user_gender = eval(input("請輸入您的性別（1.男  2.女）：")) - 1
user_age = eval(input("請輸入您的年齡（6-15）："))
user_height = eval(input("請輸入您的身高（cm）："))
user_weight = eval(input("請輸入您的體重（kg）："))

# In[]
# 訓練出四個「簡單線性迴歸器」：
# regressor = [[SimpleRegressor(), SimpleRegressor()], [SimpleRegressor(), SimpleRegressor()]]
# regressor[0][0] -->年齡 vs. 男生身高
# regressor[0][1] -->年齡 vs. 女生身高
# regressor[1][0] -->年齡 vs. 男生體重
# regressor[1][1] -->年齡 vs. 女生體重
# In[]
# regressor[0][user_gender].predict(pd.DataFrame([[user_age]]))
# pd.DataFrame([[user_age]])
# 可用下列方法，取得使用者同齡之平均身高、體重：
# h_avg = regressor[0][user_gender].predict(x_test=pd.DataFrame([[age]])).iloc[0, 0]
# w_avg = regressor[1][user_gender].predict(x_test=pd.DataFrame([[age]])).iloc[0, 0]

# 預設使用者輸入值
# user_gender=1
# user_age=10
# user_height=138
# user_weight=45

# Dataset Loading
ds_Student_Height = pp.dataset("Student_Height.csv")
ds_Student_Weight = pp.dataset("Student_Weight.csv")

# In[] Independent/Dependent Variables Decomposition
H_Years, H_All_Average = pp.decomposition(ds_Student_Height, [1],[2])
H_Man, H_Woman = pp.decomposition(ds_Student_Height, [3],[4])
W_Years, W_All_Average = pp.decomposition(ds_Student_Weight, [1],[2])
W_Man, W_Woman = pp.decomposition(ds_Student_Weight, [3],[4])

# In[] Split Training vs. Testing Set
# 男生身高
H_Man_Years_train, H_Man_Years_test, H_Man_train, H_Man_test = pp.split_train_test(H_Years, H_Man, train_size=0.8)
# 男生體重
W_Man_Years_train, W_Man_Years_test, W_Man_train, W_Man_test = pp.split_train_test(W_Years, W_Man, train_size=0.8)

# 女生身高
H_Woman_Years_train, H_Woman_Years_test, H_Woman_train, H_Woman_test = pp.split_train_test(H_Years, H_Woman, train_size=0.8)
# 女生體重
W_Woman_Years_train, W_Woman_Years_test, W_Woman_train, W_Woman_test = pp.split_train_test(W_Years, W_Woman, train_size=0.8)

# In[] Fitting Simple Regressor with HappyML
from HappyML.regression import SimpleRegressor

# 訓練出四個「簡單線性迴歸器」：
# regressor = [[SimpleRegressor(), SimpleRegressor()], [SimpleRegressor(), SimpleRegressor()]]
# regressor[0][0] -->年齡 vs. 男生身高
# regressor[0][1] -->年齡 vs. 女生身高
# regressor[1][0] -->年齡 vs. 男生體重
# regressor[1][1] -->年齡 vs. 女生體重

regressor = [[SimpleRegressor(), SimpleRegressor()], [SimpleRegressor(), SimpleRegressor()]]


# 男生
if user_gender == 0:
    # 男生身高
    regressor[0][user_gender]=regressor[0][user_gender].fit(H_Man_Years_train, H_Man_train).predict(H_Man_Years_test)
    #print(regressor[0][user_gender]) #印出身高預測值
    # 男生體重
    regressor[1][user_gender]=regressor[1][user_gender].fit(W_Man_Years_train, W_Man_train).predict(W_Man_Years_test)
    #print(regressor[1][user_gender]) #印出體種預測值


    # 計算同年齡層男生平均身高
    nday_h_man_test = pd.DataFrame(H_Man_test).to_numpy() # DataFrame to ndArray
    nday_h_man_years_test = pd.DataFrame(H_Man_Years_test).to_numpy() # DataFrame to ndArray
    # print(nday_h_man_test.size)
    iSum_h_man=0 # 累加身高
    icount_h_man=0 # 累加符合同年齡層人數
    sResult_h_man="" # 顯示結果
    for i in range(len(nday_h_man_test)):
        if nday_h_man_years_test [i][0] == user_age:  # 自變數測試集符合user年齡
            iSum_h_man=iSum_h_man + nday_h_man_test [i][0]
            #print(nday_h_man_test [i][0], i) # 印出符合同年齡層身高
            icount_h_man= icount_h_man + 1
            
    if int(iSum_h_man)==0: # 未找到同年齡符合資料
        sResult_h_man= "未找到 " + str(user_age) + " 歲平均身高資料!，您的身高為 " + str(user_height) + " 公分"
        print(sResult_h_man)
    else:
        if int(iSum_h_man) > 0: # 已找到同年齡符合資料，並做計算處理
            sResult_h_man = str(user_age) + " 歲男生平均身高為 " + str(round(iSum_h_man/icount_h_man,2)) + " 公分，您的身高為 " + str(user_height) + " 公分"
            print(sResult_h_man) # 印出結果


    # 計算同年齡層男生平均體重
    nday_w_man_test = pd.DataFrame(W_Man_test).to_numpy() # DataFrame to ndArray
    nday_w_man_years_test = pd.DataFrame(W_Man_Years_test).to_numpy() # DataFrame to ndArray
    # print(nday_w_man_test.size)
    iSum_w_man=0 # 累加身高
    icount_w_man=0 # 累加符合同年齡層人數
    sResult_w_man="" # 顯示結果
    for i in range(len(nday_w_man_test)):
        if nday_w_man_years_test [i][0] == user_age:  # 自變數測試集符合user年齡
            iSum_w_man=iSum_w_man + nday_w_man_test [i][0]
            #print(nday_w_man_test [i][0], i) # 印出符合同年齡層體重
            icount_w_man= icount_w_man +1

    if int(iSum_w_man)==0: # 未找到同年齡符合資料
        sResult_w_man= "未找到 " + str(user_age) + " 歲平均體重資料!，您的體重為 " + str(user_weight) + " 公斤"
        print(sResult_w_man)
    else:
        if int(iSum_w_man) > 0: # 已找到同年齡符合資料，並做計算處理
            sResult_w_man = str(user_age) + " 歲男生平均體重為 " + str(round(iSum_w_man/icount_w_man,2)) + " 公斤，您的體重為 " + str(user_weight) + " 公斤"
            print(sResult_w_man) # 印出結果
            
    # Visualize the Model
    import HappyML.model_drawer as md
    #sample_data -> 樣本點
    # md.sample_model(sample_data=(SH_Years_test, SH_Man_test), model_data=(SH_Years_test, Y_pred),
    #                 title="測試集樣本點 vs. 預測模型", font="DFKai-sb") #測試結果

    #sample_data(年齡(X軸),身高/體重(Y軸))
    #print(regressor[0][user_gender])
    #男生身高
    md.sample_model(sample_data=(user_age, user_height), model_data=(H_Man_Years_test, regressor[0][user_gender]),
                    title="身高落點模型", font="DFKai-sb",xlabel=sResult_h_man) #測試結果
    #男生體重
    md.sample_model(sample_data=(user_age, user_weight), model_data=(W_Man_Years_test, regressor[1][user_gender]),
                    title="體重落點模型", font="DFKai-sb",xlabel=sResult_w_man) #測試結果

else: # 女生
    # 女生身高
    regressor[0][user_gender]=regressor[0][user_gender].fit(H_Woman_Years_train, H_Woman_train).predict(H_Woman_Years_test)
    #print(regressor[0][user_gender]) #印出身高預測值
    # 女生體重
    regressor[1][user_gender]=regressor[1][user_gender].fit(W_Woman_Years_train, W_Woman_train).predict(W_Woman_Years_test)
    #print(regressor[1][user_gender]) #印出體重預測值


    # 計算同年齡層女生平均身高
    nday_h_woman_test = pd.DataFrame(H_Woman_test).to_numpy() # DataFrame to ndArray
    nday_h_woman_years_test = pd.DataFrame(H_Woman_Years_test).to_numpy() # DataFrame to ndArray
    # print(nday_h_man_test.size)
    iSum_h_woman=0 # 累加身高
    icount_h_woman=0 # 累加符合同年齡層人數
    sResult_h_man="" # 顯示結果
    for i in range(len(nday_h_woman_test)):
        if nday_h_woman_years_test [i][0] == user_age: # 自變數測試集符合user年齡
            iSum_h_woman = iSum_h_woman + nday_h_woman_test [i][0]
            #print(nday_h_woman_test [i][0], i) # 印出符合同年齡層身高
            icount_h_woman = icount_h_woman + 1
            
    if int(iSum_h_woman)==0: # 未找到同年齡符合資料
        sResult_h_woman= "未找到 " + str(user_age) + " 歲平均身高資料!，您的身高為 " + str(user_height) + " 公分"
        print(sResult_h_woman)
    else:
        if int(iSum_h_woman) > 0: # 已找到同年齡符合資料，並做計算處理
            sResult_h_woman = str(user_age) + " 歲女生平均身高為 " + str(round(iSum_h_woman/icount_h_woman,2)) + " 公分，您的身高為 " + str(user_height) + " 公分"
            print(sResult_h_woman) # 印出結果


    # 計算同年齡層女生平均體重
    nday_w_woman_test = pd.DataFrame(W_Woman_test).to_numpy() # DataFrame to ndArray
    nday_w_woman_years_test = pd.DataFrame(W_Woman_Years_test).to_numpy() # DataFrame to ndArray
    # print(nday_w_man_test.size)
    iSum_w_woman=0 # 累加身高
    icount_w_woman=0 # 累加符合同年齡層人數
    sResult_w_woman="" # 顯示結果
    for i in range(len(nday_w_woman_test)):
        if nday_w_woman_years_test [i][0] == user_age:
            iSum_w_woman=iSum_w_woman + nday_w_woman_test [i][0]
            #print(nday_w_woman_test [i][0], i) # 印出符合同年齡層體重
            icount_w_woman= icount_w_woman +1

    if int(iSum_w_woman)==0: # 未找到同年齡符合資料
        sResult_w_woman= "未找到 " + str(user_age) + " 歲平均體重資料!，您的體重為 " + str(user_weight) + " 公斤"
        print(sResult_w_woman)
    else:
        if int(iSum_w_woman) > 0: # 已找到同年齡符合資料，並做計算處理
            sResult_w_woman = str(user_age) + " 歲女生平均體重為 " + str(round(iSum_w_woman/icount_w_woman,2)) + " 公斤，您的體重為 " + str(user_weight) + " 公斤"
            print(sResult_w_woman) # 印出結果
            
    # Visualize the Model
    import HappyML.model_drawer as md
    #sample_data -> 樣本點
    # md.sample_model(sample_data=(SH_Years_test, SH_Man_test), model_data=(SH_Years_test, Y_pred),
    #                 title="測試集樣本點 vs. 預測模型", font="DFKai-sb") #測試結果

    #sample_data(年齡(X軸),身高/體重(Y軸))
    #print(regressor[0][user_gender])
    #女生身高
    md.sample_model(sample_data=(user_age, user_height), model_data=(H_Woman_Years_test, regressor[0][user_gender]),
                    title="身高落點模型", font="DFKai-sb",xlabel=sResult_h_woman) #測試結果
    #女生體重
    md.sample_model(sample_data=(user_age, user_weight), model_data=(W_Woman_Years_test, regressor[1][user_gender]),
                    title="體重落點模型", font="DFKai-sb",xlabel=sResult_w_woman) #測試結果
