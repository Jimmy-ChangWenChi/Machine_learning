# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:23:17 2024

@author: jimmy
"""
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf #抓取台股歷史資料
import dateutil.parser as psr # 萬能日期解析器


stockIdMap = pd.read_csv("TaiwanStockID.csv")
#print(file["StockName"])

stockInput = input("請輸入股票名稱或股票代號: ")
#print(type(StockInput))

#抓取股票代號
# =============================================================================
if stockInput.isdigit() :
    #print("Stcok is number")
    queryID = "{}.TW".format(stockInput)
    #print(queryID)
    condition = stockIdMap["StockID"] == eval(stockInput) # StockID值 為int, 不是整數
    #print(condition)
    stockName = stockIdMap[condition].iloc[0]["StockName"]
else:
    #print("Stock is string")
    stockName = stockInput
    condition = stockIdMap["StockName"] == stockInput
    stockId = stockIdMap[condition].iloc[0]["StockID"]
    queryID = "{}.TW".format(stockId)
# =============================================================================



# =============================================================================
Start_date = input("請輸入起始日期: ")
End_date = input("請輸入中止日期: ")
# =============================================================================
#解析時間格式
# =============================================================================
startDate = psr.parse(Start_date)
EndDate = psr.parse(End_date)
# print(startDate)
# print(EndDate)

startDate.strftime("%Y-%m-%d") #轉成要的格式
EndDate.strftime("%Y-%m-%d")
# print(startDate.strftime("%Y-%m-%d"))
# print(EndDate.strftime("%Y-%m-%d"))


#抓取資料
# =============================================================================
data = yf.download(queryID,start=startDate.strftime("%Y-%m-%d"),end=EndDate.strftime("%Y-%m-%d"))
# print(data)
# =============================================================================


#建立收盤價
close_price = data["Close"]
#close_price.plot(label = "收盤價")
# =============================================================================

#建立20天
closePrice_20Day = close_price.rolling(window = 20) #20日壘加總合
closePriceMean_20Day = closePrice_20Day.mean()  #20日壘加總合平均
#closePriceMean_20Day.plot(label="20MA")
# =============================================================================

#建立60天
closePrice_60Day = close_price.rolling(window = 60) 
closePriceMean_60Day = closePrice_60Day.mean()  
#closePriceMean_60Day.plot(label="60MA")
# =============================================================================


#繪圖

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False
title = stockName + ' ' + startDate.strftime("%Y-%m-%d") + " ~ " + EndDate.strftime("%Y-%m-%d") + " 收盤價"
plt.title(title)
plt.xlabel("Date") # Y軸
plt.ylabel("指數") # Y軸

plt.plot(close_price,label="收盤價")
plt.plot(closePriceMean_20Day,label="20MA")
plt.plot(closePriceMean_60Day,label="60MA")
plt.legend(loc="best") # 圖例位置 
plt.show()
# =============================================================================



# =============================================================================
#data = yf.download("2330.TW",start="2022-01-01",end="2022-12-31") #抓取資料
#print(data)

# =============================================================================
# close_price = data["Close"]
# High_price = data["High"]
# close_price.plot(label="收盤價")
# High_price.plot(label="最高點")
# =============================================================================


# =============================================================================
# 
# data = yf.download("2330.TW",start="2022-01-01",end="2022-12-31") #抓取資料
# 
# print(data)
# 
# 
# close_price = data["Close"] #抓收盤價
# print(close_price)
# 
# close_price.plot(label="收盤價") #繪製圖形
# 
# file = pd.read_csv("TaiwanStockID.csv")
# print(file)
# =============================================================================
