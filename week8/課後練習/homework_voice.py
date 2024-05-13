# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:22:30 2024

@author: jimmy
"""

import HappyML.preprocessor as pp
from HappyML.preprocessor import KBestSelector
from HappyML.classification import SVM
from HappyML.performance import KFoldClassificationPerformance

dataset = pp.dataset(file="Voice.csv")

X, Y = pp.decomposition(dataset, x_columns=[i for i in range(0, 19)], y_columns=[20])

Y, Y_mapping = pp.label_encoder(Y,mapping=True)

import HappyML.preprocessor as pp

selector = KBestSelector(best_k = 'auto')

X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort=True).transform(x_ary=X)

X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y)



classifier = SVM()
Y_pre_svm = classifier.fit(X_train,Y_train).predict(X_test)



K = 10
kfp = KFoldClassificationPerformance(x_ary= X, y_ary=Y, classifier= classifier.classifier, k_fold=  K)

print("----- SVM Classification(No GridSearch)-----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))



#GridSearch start

X, Y = pp.decomposition(dataset, x_columns=[i for i in range(0, 19)], y_columns=[20])
Y, Y_mapping = pp.label_encoder(Y,mapping=True)
selector = KBestSelector(best_k='auto')
X = selector.fit(x_ary = X, y_ary = Y, verbose = True, sort=True).transform(x_ary=X)
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y)


#prepare GridSearch parameters
import numpy as np
C_range = np.logspace(0, 3, 4) # Create [1, 10, 100, 1000]
Gamma_range = np.logspace(-4, -1, 4) # Create [0.0001, 0.001, 0.01, 0.1]
Coef0_range = np.logspace(0, 3, 4) # Create [1, 10, 100, 1000]

# Combination of Hyper Parameters
Linear_dict = dict(kernel=["linear"], C=C_range, coef0=Coef0_range)
RBF_dict = dict(kernel=["rbf"], C=C_range, gamma=Gamma_range)
Sigmoid_dict = dict(kernel=["sigmoid"], C=C_range, gamma=Gamma_range, coef0=Coef0_range)
#Poly_dict = dict(kernel=["poly"],C=C_range,gamma=Gamma_range,coef0=Coef0_range)

# Collect all Combinations for Grid Search
params_list = [Linear_dict, RBF_dict, Sigmoid_dict]

classifier = SVM()

from HappyML.performance import GridSearch
validator = GridSearch(estimator=classifier.classifier, parameters=params_list, verbose=True)
validator.fit(x_ary=X, y_ary=Y)

print("Best Parameters: {}  Best Score: {}".format(validator.best_parameters, validator.best_score))
classifier.classifier = validator.best_estimator


Y_pred_svm = classifier.fit(X_train, Y_train).predict(X_test)

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- SVM Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

