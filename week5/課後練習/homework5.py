# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:42:03 2024

@author: jimmy
"""

# In[] Pre-processing
import HappyML.preprocessor as pp

# Dataset Loading
dataset = pp.dataset("Insurance.csv")

# Independent/Dependent Variables Decomposition
X, Y = pp.decomposition(dataset, [0, 1, 2, 3, 4, 5], [6])

# Apply One Hot Encoder to Column[3] & Remove Dummy Variable Trap
X = pp.onehot_encoder(X, columns=[1,4,5], remove_trap=True)
#X = pp.onehot_encoder(X,columns=[1,4,5])

# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

# Feature Scaling (optional)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

# In[] Multiple Linear Regression with HappyML 多元線性回歸
from HappyML.regression import MultipleRegressor

model = MultipleRegressor()
#selected_features = model.backward_elimination(X_train, Y_train, verbose=True)
model.backward_elimination(X_train, Y_train, verbose=True) #反向淘汰法
Y_pred = model.fit(X_train, Y_train).predict(X_test)

print("Adjusted R-Squared:", model.r_score())

# # In[] Cross-model Performance 用 "殘差" 來決定效能
from HappyML.performance import rmse #均方根誤差

rmse_multi = rmse(Y_test, Y_pred)

# In[] Check for Assumption of Regression
from HappyML.criteria import AssumptionChecker

X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
Y_train, Y_test, Y_pred = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test, Y_pred))

checker = AssumptionChecker(X_train.loc[:, model.named_features], X_test.loc[:, model.named_features], Y_train, Y_test, Y_pred)
checker.y_lim = (-4, 4)
checker.heatmap = True
checker.check_all()