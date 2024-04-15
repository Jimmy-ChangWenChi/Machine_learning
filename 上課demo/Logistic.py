#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:06:21 2024

@author: jimmy
"""

import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Social_Network_Ads.csv")

# X, Y Decomposition
X, Y = pp.decomposition(dataset, x_columns=[1, 2, 3], y_columns=[4])

# Categorical Data Encoding & Remove Dummy Variable Trap
X = pp.onehot_encoder(X, columns=[0], remove_trap=True)

# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

from sklearn.linear_model import LogisticRegression
import time

# Model Creation
model = LogisticRegression(solver="lbfgs", random_state=int(time.time()))

# Training
model.fit(X_train, Y_train.values.ravel())

# Testing
Y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

print("Confusion Matrix:\n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2%}")
print(f"Recall: {recall_score(Y_test, Y_pred):.2%}")
print(f"Precision: {precision_score(Y_test, Y_pred):.2%}")
print(f"F1-score: {fbeta_score(Y_test, Y_pred, beta=1):.2%}")