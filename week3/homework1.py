# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:28:34 2024

@author: jimmy
"""

import numpy as np
dice = [1,2,3,4,5,6]
value = [0.1,0.1,0.2,0.1,0.2,0.3]

Thesis_result = dict(zip(dice,value)) #合成字典
#print(Thesis_result)

#產生不規則骰子 100次結果
#result = np.random.choice(dice,[100],replace=True,p = [0.1,0.1,0.2,0.1,0.2,0.3])
result = np.random.choice(dice,[100],replace=True,p = value)
#print(result)

#計算實際樣本數
dice1, value1 = np.unique(result,return_counts=True) #return_counts=True 不重複樣本數 出現次數
Real_result = dict(zip(dice1,value1/100))

#print(dice1,value1)
#print(dict(zip(dice1,value1/100))) 

print(result)
print(Thesis_result)
print(Real_result)


