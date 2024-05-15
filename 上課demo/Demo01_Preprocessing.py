# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:44:17 2019

@author: 俊男
"""

# In[] Import & Load Data Set
import numpy as np
import pandas as pd

dataset = pd.read_csv("CarEvaluation.csv")

# In[] Decomposition the dataset into Independent & Dependent Variables
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# In[] Missing Data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:, 1:4])

# In[] Categorcal Data, Label Encoder
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(Y).astype("float64")

# In[] Categorical Data, One Hot Encoder

ary_dummies = pd.get_dummies(X[:, 0]).values
X = np.concatenate((ary_dummies, X[:, 1:4]), axis=1).astype("float64")

# In[] Training Set vs. Testing Set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# In[] Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

