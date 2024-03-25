# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:00:03 2019

@author: 俊男
"""

salary = eval(input("請輸入您的月薪："))

if salary <= 0:
    print("您的薪資是：{}".format(salary))
else:
    print("薪資不可以是負數！請重新輸入！")


