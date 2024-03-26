# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:47:02 2019

@author: 俊男
"""

# In[] Import Packages
import pandas as pd

# In[] Create DataFrame
Rows = ["Row1", "Row2", "Row3"]
Columns = ["Column1", "Column2"]

# by list (Row-based)
dfList = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=Rows, columns=Columns)
print(dfList)

# by Dict (Column-based)
dfDict = pd.DataFrame({"Column1":[1, 3, 5], "Column2":[2, 4, 6]}, index=Rows)
print(dfDict)

# In[] Create DataFrame by CSV file
dfCSV = pd.read_csv("CarSales.csv")
print(dfCSV)

print(dfCSV["Country"].mode())
print(dfCSV["Age"].median())
print(dfCSV["Salary"].mean())

# In[] Create DataFrame by HTML page
dfHTML = pd.read_html("http://www.stockq.org/market/asia.php")

# In[] Selection
asia_stocks = dfHTML[7].loc[2:, :5]
print(asia_stocks)

# In[] Converted to NumPy
ary = asia_stocks.to_numpy()
print(ary)

# In[] Selection by Conditions
stocks_table = dfHTML[7].loc[2:, [0, 1, 2, 4, 5]]
condition = stocks_table[2].astype("float64") > 0
print(stocks_table[condition])

# In[] Other attributes
#print(asia_stocks.shape)
#print(asia_stocks.index)
#print(asia_stocks.columns)
#print(asia_stocks.describe())
