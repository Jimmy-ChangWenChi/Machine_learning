# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:23:39 2023

@author: cnchi
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Social_Network_Ads.csv")

# X, Y Decomposition
X, Y = pp.decomposition(dataset, x_columns=[1, 2, 3], y_columns=[4])

# Categorical Data Encoding & Remove Dummy Variable Trap
X = pp.onehot_encoder(X, columns=[0], remove_trap=True)

# # Feature Selection (wiht Standard Library)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# kbest = SelectKBest(score_func=chi2, k=2)
# kbest = kbest.fit(X, Y)
# print(f"The p-values of Feature Importance: {kbest.pvalues_}")

# X = kbest.transform(X)

# Feature Selection (with HappyML)
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training & Testing set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Logistic Regression (Standard Library)
# from sklearn.linear_model import LogisticRegression
# import time

# # Model Creation
# model = LogisticRegression(solver="lbfgs", random_state=int(time.time()))

# # Training
# model.fit(X_train, Y_train.values.ravel())

# # Testing
# Y_pred = model.predict(X_test)

# In[] Logistic Regression (HappyML)
from HappyML.regression import LogisticRegressor

# Training & Predict
model = LogisticRegressor()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

# In[] Evaluation
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import fbeta_score

# print("Confusion Matrix:\n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))
# print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2%}")
# print(f"Recall: {recall_score(Y_test, Y_pred):.2%}")
# print(f"Precision: {precision_score(Y_test, Y_pred):.2%}")
# print(f"F1-score: {fbeta_score(Y_test, Y_pred, beta=1):.2%}")

# In[] Evaluation (HappyML)
from HappyML.performance import ClassificationPerformance

pfm = ClassificationPerformance(Y_test, Y_pred)

print("Confusion Matrix:\n", pfm.confusion_matrix())
print(f"Accuracy: {pfm.accuracy():.2%}")
print(f"Recall: {pfm.recall():.2%}")
print(f"Precision: {pfm.precision():.2%}")
print(f"F1-score: {pfm.f_score():.2%}")

# In[] Visualize the Result
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=model.regressor, 
                   title="訓練集樣本點 vs. 模型", font="DFKai-sb")
md.classify_result(x=X_test, y=Y_test, classifier=model.regressor, 
                   title="測試集樣本點 vs. 模型", font="DFKai-sb")
