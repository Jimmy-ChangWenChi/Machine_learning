# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:10:12 2019

@author: 俊男
"""

# In[] Pre-processing
from HappyML import preprocessor as pp

# Dataset Loading
dataset = pp.dataset("Salary_Data.csv")

# Independent/Dependent Variables Decomposition
X, Y = pp.decomposition(dataset, [0], [1])

# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=2/3)

# Feature Scaling (optional)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

# In[] Fitting Simple Regressor
# from sklearn.linear_model import LinearRegression

# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_pred = regressor.predict(X_test)

# R_Score = regressor.score(X_test, Y_test)

# In[] Fitting Simple Regressor with HappyML
from HappyML.regression import SimpleRegressor

regressor = SimpleRegressor()
Y_pred = regressor.fit(X_train, Y_train).predict(X_test)
print("R-Squared Score:", regressor.r_score(X_test, Y_test))

# In[] Visualize the Model
import HappyML.model_drawer as md

sample_data=(X_train, Y_train)
model_data=(X_train, regressor.predict(X_train))
md.sample_model(sample_data=sample_data, model_data=model_data,
                title="訓練集樣本點 vs. 預測模型", font="DFKai-sb")
md.sample_model(sample_data=(X_test, Y_test), model_data=(X_test, Y_pred),
                title="測試集樣本點 vs. 預測模型", font="DFKai-sb")

# In[] Assumption Checking

from HappyML.criteria import AssumptionChecker

checker = AssumptionChecker(X_train, X_test, Y_train, Y_test, Y_pred)
# checker.sample_linearity()
# checker.residuals_normality()
# checker.residuals_independence()
# checker.residuals_homoscedasticity(y_lim=(-4, 4))
# checker.features_correlation(heatmap=True)
checker.y_lim = (-4, 4)
checker.heatmap = True
checker.check_all()
