# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:08:21 2024

@author: jimmy
"""

import pandas as pd

#建立dataframe

#透過 複合資料結構 建立

Rows = ["Row1","Row2","Row3"]
Colums = ["Column1","Column2","Columns3"]

#by list
# =============================================================================
# dfList = pd.DataFrame([[1,2],[3,4],[5,6,7]], index = Rows,columns = Colums)
# #dfList = pd.DataFrame([[1,2],[3,4],[5,6]])
# print(dfList)
# =============================================================================

#by Dict
# =============================================================================
# dfDict = pd.DataFrame({"Columns1":[1,2,3],"Columns2":[4,5,6],"Columns3":[7,8,9]}, index = Rows)
# print(dfDict)
# 
# =============================================================================

#透過 CSV檔 建立
# =============================================================================
# dfCSV = pd.read_csv("CarSales.csv")
# print(dfCSV)
# 
# print(dfCSV["Country"].mode()) # .mode()取眾數
# print(dfCSV["Age"].median()) # .median() 取中位數
# print(dfCSV["Salary"].mean()) # .mean() 取平均值
# 
# =============================================================================

# =============================================================================
# #透過HTML 建立
# dfHTML = pd.read_html("http://www.stockq.org/market/asia.php")
# print(dfHTML)
# 
# #選取特定欄位
# 
# asia_stocks = dfHTML[9].loc[2:,:"亞洲股市行情 (Asian Markets)"] # 9為dfHTML的dataFrame欄位
# #asia_stocks = dfHTML[9].loc[2:,:7]
# print(asia_stocks)
# """
# loc[] = Locator
# 使用 欄位名稱 抓取資料
# 索引值上限 = 包含該索引值
# 7 為欄位名稱
# 
# iloc[] = index Locator
# 使用 索引值 抓取資料
# 索引值上限 = 不包含該索引值
# """
# asia_stocks = dfHTML[9].iloc[2:,:7]
# print(asia_stocks)
# =============================================================================

#pandas 轉為 Numpy陣列
# =============================================================================
# dfHTML = pd.read_html("http://www.stockq.org/market/asia.php")
# asia_stocks = dfHTML[9].loc[2:,:5]
# 
# ary = asia_stocks.to_numpy()
# #ary = asia_stocks.values
# print(ary)
# =============================================================================

#以 條件 過濾DataFrame

dfHTML = pd.read_html("http://www.stockq.org/market/asia.php")
asia_stocks = dfHTML[9].loc[2:,:5]

condition = asia_stocks[2].astype("float64") > 10
print(asia_stocks[condition])





