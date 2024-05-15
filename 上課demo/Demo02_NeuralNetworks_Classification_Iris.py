# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 07:22:01 2019

@author: 俊男
"""

# In[] Data Pre-processing
from sklearn.datasets import load_iris
import HappyML.preprocessor as pp

# Load Data
dataset = load_iris()

# X, Y
import pandas as pd
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
Y = pd.DataFrame(dataset.target, columns=["Iris_Type"])
Y_name = dataset.target_names.tolist()

# One_hot incoder for Y
Y = pp.onehot_encoder(ary=Y, columns=[0])

# Feature Selection
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k="auto")
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Neural Networks without HappyML's Class
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the whole Neural Networks
classifier = Sequential()

# Add the Input & First Hidden Layer
classifier.add(Dense(input_dim=X_train.shape[1], units=20, kernel_initializer="random_normal", activation="relu"))

# Add the Second Hidden Layer
classifier.add(Dense(units=10, kernel_initializer="random_normal", activation="relu"))

# Add the Output Layer
classifier.add(Dense(units=3, kernel_initializer="random_normal", activation="softmax"))

# Compile the whole Neural Networks
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit
classifier.fit(x=X_train, y=Y_train, batch_size=10, epochs=100)

# Predict
import pandas as pd
Y_pred = classifier.predict(x=X_test)
Y_pred = (Y_pred > 0.5).astype(int)
Y_pred = pd.DataFrame(Y_pred, index=Y_test.index).idxmax(axis=1)

Y_test.columns = [0, 1, 2]
Y_test = Y_test.idxmax(axis=1)

# In[] Performance
from HappyML.performance import ClassificationPerformance

pfm = ClassificationPerformance(Y_test, Y_pred)

print("Confusion Matrix:\n", pfm.confusion_matrix())
print("Accuracy: {:.2%}".format(pfm.accuracy()))
print("Recall: {:.2%}".format(pfm.recall()))
print("Precision: {:.2%}".format(pfm.precision()))
print("F1-score: {:.2%}".format(pfm.f_score()))