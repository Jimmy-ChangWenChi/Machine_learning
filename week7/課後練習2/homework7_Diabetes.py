# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:04:43 2024

@author: jimmy
"""

import HappyML.preprocessor as pp

dataset = pp.dataset(file="Diabetes.csv")


X, Y = pp.decomposition(dataset, x_columns=[0,1,2,3,4,5,6,7], y_columns=[8])

from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

from HappyML.classification import NaiveBayesClassifier

model = NaiveBayesClassifier()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=model.classifier, k_fold=K, verbose=False)

#print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print(f"{K} Folds Mean Accuracy: {kfp.accuracy():.2%}")
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))


#檢查 自變數各自獨立
from HappyML.criteria import AssumptionChecker
checker = AssumptionChecker(X_train, X_test, Y_train, Y_test, Y_pred)
checker.features_correlation(heatmap=True)


#設定 K =2 的單純貝式
X, Y = pp.decomposition(dataset, x_columns=[0,1,2,3,4,5,6,7], y_columns=[8])

selector = KBestSelector(best_k=2)
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))


model_2 = NaiveBayesClassifier()
Y_pred = model_2.fit(X_train, Y_train).predict(X_test)
K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=model.classifier, k_fold=K, verbose=False)

#print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print(f"{K} Folds Mean Accuracy: {kfp.accuracy():.2%}")
print(f"{K} Folds Mean Recall: {kfp.recall():.2%}")
print(f"{K} Folds Mean Precision: {kfp.precision():.2%}")
print(f"{K} Folds Mean F1-Score: {kfp.f_score():.2%}")

import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=model_2.classifier, title="訓練集 vs. 模型", font='DFKai-sb')
md.classify_result(x=X_test, y=Y_test, classifier=model_2.classifier, title="測試集 vs. 模型", font='DFKai-sb')


#設定 K = 2 的邏輯回歸
from HappyML.regression import LogisticRegressor
X, Y = pp.decomposition(dataset, x_columns=[0,1,2,3,4,5,6,7], y_columns=[8])

selector = KBestSelector(best_k=2)
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y,train_size=0.8)
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

model_logic = LogisticRegressor()

Y_pred = model_2.fit(X_train, Y_train).predict(X_test)
from HappyML.performance import ClassificationPerformance
pfm = ClassificationPerformance(Y_test, Y_pred)

print(f"Accuracy: {pfm.accuracy():.2%}")
print(f"Recall: {pfm.recall():.2%}")
print(f"Precision: {pfm.precision():.2%}")
print(f"F1-score: {pfm.f_score():.2%}")

md.classify_result(x=X_train, y=Y_train, classifier=model_2.classifier, title="訓練集 vs. 模型", font='DFKai-sb')
md.classify_result(x=X_test, y=Y_test, classifier=model_2.classifier, title="測試集 vs. 模型", font='DFKai-sb')
