# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:23:17 2024

@author: jimmy
"""
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

Stock = input("請輸入股票 ")
print(type(Stock))

# =============================================================================
# 
# data = yf.download("2330.TW",start="2022-01-01",end="2022-12-31")
# 
# print(data)
# 
# 
# close_price = data["Close"]
# print(close_price)
# 
# close_price.plot(label="收盤價")
# 
# file = pd.read_csv("TaiwanStockID.csv")
# print(file)
# =============================================================================
