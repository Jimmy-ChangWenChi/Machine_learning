# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:24:18 2024

@author: jimmy
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# #折線圖
# # plt.plot([X軸資料],[Y軸資料],[線段格式])
# #單組資料
# # plt.plot([2,3,4,20])
# # plt.show()
# 
# #plt.plot([2,4,6,8],"r-") 
# 
# # 代號 r(red),g(green),b(blue),y(yellow),c(Cyan),m(Magenta),k(black),w(white)
# # 代號 "o"/"s"(圓點/方形),"^"/"v"/"<"/">"(三角形(上下左右)),"p"(五角形),"*"(星形),"x"(X形),"D"(菱形),
# #      "-"/"--"/"-."/"."(實線,Dash-Dash虛線,Dash-Dot虛線,Dot虛線)
#
# #解決中文是豆腐框
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
# # rc = Resource參數設定, 如Windows找不到Arial Unicode MS 改成 DFKai-sb,macOS 改成 SimHei
# plt.rcParams['axes.unicode_minus'] = False
# # 軸線上有負數,則不會用unicode的全形字,使用半形字
#
# X= np.arange(0.,5,0.2)
# plt.title("X, X*2, X*3 線圖") # 圖片標題
# plt.xlabel("X軸") # X軸
# plt.ylabel("Y軸") # Y軸
# 
# #多組資料
# plt.plot(X,X,"r--",label="X")
# plt.plot(X,X*2,"b--",label="X*2")
# plt.plot(X,X*3,"g--",label="X*3")
# plt.legend(loc="lower right") # 圖例位置 
# # loc= location, best = 最空曠, center = 正中央, upper =最上方, lower=最下方, upper跟lower 需搭配左右
# plt.show()
# =============================================================================

# =============================================================================
# 
# #長條圖
# # plt.bar(X,Y)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
# plt.rcParams['axes.unicode_minus'] = False
# 
# X= ["APPLE","SAMSUNG","OPPO","ASUS","小米","SONY", "NOKIA"]
# Y= [24.1,23.3,9.2,7.6,6.5,4.6,1.0]
# # X,Y 不對等, 無法顯示出來
# plt.title("2020年 台灣手機市占率")
# plt.xlabel("手機品牌")
# plt.ylabel("百分比")
# 
# plt.bar(X,Y)
# plt.show()
# =============================================================================

#散佈圖
# plt.scatter(X,Y,Style)

n = 150 #樣本點

X1 = np.random.normal(-1,1,n)
Y1 = np.random.normal(2,1,n)

X2 = np.random.normal(2,1,n)
Y2 = np.random.normal(-1,1,n)

plt.scatter(X1, Y1,s=75,c="red", marker = "+") #75 為75px, c為顏色
plt.scatter(X2, Y2,s=75, c="blue", marker = "s")
plt.show()