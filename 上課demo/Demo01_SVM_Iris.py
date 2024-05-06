# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:16:55 2019

@author: 俊男
"""

# In[] Data Pre-processing
from sklearn.datasets import load_iris

# Load Data
dataset = load_iris()

# X, Y
import pandas as pd
X = pd.DataFrame(dataset.data, columns=dataset.feature_names) #載入 NDArray並包覆成 DataFrame
Y = pd.DataFrame(dataset.target, columns=["Iris_Type"])
Y_name = dataset.target_names.tolist()

# Load HappyML
from HappyML.preprocessor import KBestSelector
import HappyML.preprocessor as pp

# Feature Selection
selector = KBestSelector(best_k=2)
#selector = KBestSelector(best_k='auto')
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Comparison: Naive Bayes
from HappyML.classification import NaiveBayesClassifier

clr_bayes = NaiveBayesClassifier()
Y_pred_bayes = clr_bayes.fit(X_train, Y_train).predict(X_test)

# Performance
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=clr_bayes.classifier, k_fold=K)

print("----- Naive Bayes Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# Visualization
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=clr_bayes.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="訓練集 vs. 貝氏模型", font="DFKai-sb")
md.classify_result(x=X_test, y=Y_test, classifier=clr_bayes.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="測試集 vs. 貝氏模型", font="DFKai-sb")

# In[] SVM without GridSearch

# # from sklearn.svm import SVC
# # import time

# # classifier = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=int(time.time()))
# # classifier.fit(X_train, Y_train.values.ravel())
# # Y_pred = classifier.predict(X_test)

# from HappyML.classification import SVM

# classifier = SVM()
# Y_pred_svm = classifier.fit(X_train, Y_train).predict(X_test)


# In[] SVM with HappyML & Gridsearch

# 準備Parameters --------------------
import numpy as np

# Ranges of Hyper Parameters
C_range = np.logspace(3, 6, 4) # Create [1000, 10000, 100000, 1000000]
Gamma_range = np.logspace(-4, -1, 4) # Create [0.0001, 0.001, 0.01, 0.1]
Coef0_range = np.logspace(0, 3, 4) # Create [1, 10, 100, 1000]
#Poly_range = np.logspace(start, stop,4) # Create []

# Combination of Hyper Parameters
Linear_dict = dict(kernel=["linear"], C=C_range, coef0=Coef0_range)
RBF_dict = dict(kernel=["rbf"], C=C_range, gamma=Gamma_range)
Sigmoid_dict = dict(kernel=["sigmoid"], C=C_range, gamma=Gamma_range, coef0=Coef0_range)
#Poly_dict = dict(kernel=["poly"],C=C_range,gamma=Gamma_range,coef0=Coef0_range)

# Collect all Combinations for Grid Search
params_list = [Linear_dict, RBF_dict, Sigmoid_dict]

# GridSearch --------------------
from HappyML.classification import SVM
classifier = SVM()

# GridSearch without HappyML
# from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(estimator=classifier.classifier, param_grid=params_list, verbose=10, cv=10)
# grid_search.fit(X, Y.values.ravel())

# print("Best Parameters: {}  Best Score: {}".format(grid_search.best_params_, grid_search.best_score_))
# classifier.classifier = grid_search.best_estimator_

# GridSearch with HappyML
from HappyML.performance import GridSearch

validator = GridSearch(estimator=classifier.classifier, parameters=params_list, verbose=True)
validator.fit(x_ary=X, y_ary=Y)

print("Best Parameters: {}  Best Score: {}".format(validator.best_parameters, validator.best_score))
classifier.classifier = validator.best_estimator

# Train & Predict
Y_pred_svm = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance (Note: Never try kernel="linear" of SVC() for K-fold, it will hang!!)

# Performance
K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# Visualization
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=classifier.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="訓練集 vs. SVM 模型", font="DFKai-sb")
md.classify_result(x=X_test, y=Y_test, classifier=classifier.classifier, fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"), title="測試集 vs. SVM 模型", font="DFKai-sb")
