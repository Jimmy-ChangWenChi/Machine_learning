#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:44:42 2024

@author: Jimmy
"""

import numpy as np
import pandas as pd
import HappyML.preprocessor as pp


#載入資料
dataset = pp.dataset("CarEvaluation.csv")

#切分自變數,應變數
X,Y = pp.decomposition(dataset,x_columns=[i for i in range(4)], y_columns=[4])

#補足缺失資料
X = pp.missing_data(X,strategy="mean")

#類別資料數位化
Y,Y_mapping = pp.label_encoder(Y,mapping=True)
Ｘ＝pp.onehot_encoder(X,columns=[0])

#切分訓練集 測試集
X_train,X_test,Y_train,Y_test = pp.split_train_test(X, Y,train_size=0.8,random_state=0)

#特徵縮放
X_train,X_test = pp.feature_scaling(X_train,transform_arys=(X_train,X_test))
