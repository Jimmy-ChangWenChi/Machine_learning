# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:38:01 2019

@author: 俊男
"""

# In[] Import & Load data
import HappyML.preprocessor as pp

dataset = pp.dataset(file="CarEvaluation.csv")

# In[] Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(4)], y_columns=[4])

# In[] Missing Data
X = pp.missing_data(X, strategy="mean")

# In[] Categorical Data Encoding

# Label Encoding
Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# One-Hot Encoding
X = pp.onehot_encoder(X, columns=[0])

# In[] Split Training Set, Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8, random_state=0)

# In[] Feature Scaling for X_train, X_test
X_train, X_test = pp.feature_scaling(X_train, transform_arys=(X_train, X_test))
