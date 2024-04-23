# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:03:57 2024

@author: jimmy
"""

import HappyML.preprocessor as pp
from sklearn.datasets import load_breast_cancer
import pandas as pd

# K = 
dataset= load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
Y = pd.DataFrame(dataset.target, columns=["isBreastCancer"])

from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y,train_size=0.8)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

from HappyML.regression import LogisticRegressor

# Training & Predict
model = LogisticRegressor()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

from HappyML.performance import ClassificationPerformance
pfm = ClassificationPerformance(Y_test, Y_pred)

print("Confusion Matrix:\n", pfm.confusion_matrix())
print(f"Accuracy: {pfm.accuracy():.2%}")
print(f"Recall: {pfm.recall():.2%}")
print(f"Precision: {pfm.precision():.2%}")
print(f"F1-score: {pfm.f_score():.2%}")



#設定K=2
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
Y = pd.DataFrame(dataset.target, columns=["isBreastCancer"])

from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k=2)
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y,train_size=0.8)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

model_2 = LogisticRegressor()
Y_pred = model_2.fit(X_train, Y_train).predict(X_test)

# import HappyML.model_drawer as md
# md.classify_result(x=X_train, y=Y_train, classifier=model_2.regressor,
#                     title="訓練集樣本點 vs. 模型", font="DFKai-sb")
# md.classify_result(x=X_test, y=Y_test, classifier=model_2.regressor, title="測試集樣本點 vs. 模型", font="DFKai-sb")

