# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 08:37:09 2023

@author: cnchi
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Position_Salaries.csv")

# Decomposition of Variables
X, Y = pp.decomposition(dataset, x_columns=[1], y_columns=[2])

# Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

# In[] Test for Polynomial Features
# from sklearn.preprocessing import PolynomialFeatures
# import pandas as pd

# # Add the X-squared feature
# poly_feat = PolynomialFeatures(degree=4)
# X_poly = pd.DataFrame(poly_feat.fit_transform(X))

# # Training & Predict with Polynomial Features
# from HappyML.regression import SimpleRegressor
# import HappyML.model_drawer as md

# model = SimpleRegressor()
# model.fit(X_poly, Y)
# Y_pred = model.predict(X_poly)

# md.sample_model(sample_data=(X, Y), model_data=(X, Y_pred))

# In[] Find Optimal Polynomial Features
# from sklearn.preprocessing import PolynomialFeatures
# import pandas as pd
# from HappyML.regression import SimpleRegressor
# import HappyML.model_drawer as md
# from HappyML.performance import rmse

# deg = 5
# poly_feat = PolynomialFeatures(degree=deg)
# X_poly = pd.DataFrame(poly_feat.fit_transform(X))

# X_train, X_test, Y_train, Y_test = pp.split_train_test(X_poly, Y, train_size=0.8)

# model = SimpleRegressor()
# model.fit(X_train, Y_train)
# Y_pred = model.predict(X_test)

# md.sample_model(sample_data=(X, Y), model_data=(X, model.predict(X_poly)))
# print(f"Degree={deg}  RMSE={rmse(Y_test, Y_pred):.4f}")

# In[] Polynomial Regression with HappyML's Class
from HappyML.regression import PolynomialRegressor
import HappyML.model_drawer as md

model = PolynomialRegressor()
model.best_degree(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, verbose=True)
Y_pred = model.fit(x_train=X, y_train=Y).predict(x_test=X)

md.sample_model(sample_data=(X, Y), model_data=(X, Y_pred))
print(f"Fit completed! The best degree is {model.degree}")
