# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:38:29 2019

@author: 俊男
"""

# In[] Pre-processing
import HappyML.preprocessor as pp

# Dataset Loading
dataset = pp.dataset("50_Startups.csv")

# Independent/Dependent Variables Decomposition
X, Y = pp.decomposition(dataset, [0, 1, 2, 3], [4])

# Apply One Hot Encoder to Column[3]
X = pp.onehot_encoder(X, columns=[3], remove_trap=True)

# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

# Feature Scaling (optional)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

# In[] Neural Networks without HappyML's Class
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the whole Neural Networks
regressor = Sequential()

# Add the Input & First Hidden Layer
regressor.add(Dense(input_dim=X_train.shape[1], units=3, kernel_initializer="normal", activation="relu"))

# Add the Second Hidden Layer
regressor.add(Dense(units=2, kernel_initializer="normal", activation="relu"))

# Add the Output Layer
regressor.add(Dense(units=1, kernel_initializer="normal", activation="linear"))

# Compile the whole Neural Networks
regressor.compile(optimizer="adam", loss="mse", metrics=["mse"])

# Fit
regressor.fit(x=X_train, y=Y_train, batch_size=5, epochs=50)

# Predict
import pandas as pd
Y_pred = pd.DataFrame(regressor.predict(x=X_test), index=Y_test.index, columns=Y_test.columns)

# In[] Performance with RMSE
from HappyML.performance import rmse

print("The RMSE of Neural Networks: {:.4f}".format(rmse(Y_test, Y_pred)))