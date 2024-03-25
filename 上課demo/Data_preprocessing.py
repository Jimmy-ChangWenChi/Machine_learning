#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:17:15 2024

@author: Jimmy
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("CarEvaluation.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values


imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:,1:4])
X[:,1:4] = imputer.transform(X[:,1:4])

labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(Y).astype("float64")

ary_dummies = pd.get_dummies(X[:,0]).values
X = np.concatenate((ary_dummies,X[:,1:4]),axis = 1).astype("float64")

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

