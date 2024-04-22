# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:56:46 2023

@author: cnchi
"""

# In[] Data Preprocessing
import HappyML.preprocessor as pp

# Load Data
dataset = pp.dataset(file="Social_Network_Ads.csv")

# X, Y decomposition
X, Y = pp.decomposition(dataset, x_columns=[1, 2, 3], y_columns=[4])

# One-Hot Encoder
X = pp.onehot_encoder(ary=X, columns=[0], remove_trap=True)

# Feature Selection
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / TEsting Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# Feature Scaling
X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Naive Bayes with Standard Libraries
# from sklearn.naive_bayes import GaussianNB

# model = GaussianNB()
# model.fit(X_train, Y_train.values.ravel())

# Y_pred = model.predict(X_test)

# In[] Naive Bayes with HappyML
from HappyML.classification import NaiveBayesClassifier

model = NaiveBayesClassifier()
Y_pred = model.fit(X_train, Y_train).predict(X_test)

# In[] K-Fold Cross Validation with Standard Libraries
# from sklearn.model_selection import cross_val_score

# k_fold = 10
# accuracies = cross_val_score(estimator=model.classifier, X=X, y=Y.values.ravel(), scoring="accuracy", cv=k_fold, n_jobs=-1)
# #print("{} Folds Mean Accuracy: {}".format(k_fold, accuracies.mean()))
# print(f"{k_fold} Folds Mean Accuracy: {accuracies.mean()}")
# print(f"{k_fold} Folds Mean Accuracy: {accuracies.mean():.2f}")

# recalls = cross_val_score(estimator=model.classifier, X=X, y=Y.values.ravel(), scoring="recall", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean Recall: {}".format(k_fold, recalls.mean()))

# precisions = cross_val_score(estimator=model.classifier, X=X, y=Y.values.ravel(), scoring="precision", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean Precision: {}".format(k_fold, precisions.mean()))

# f_scores = cross_val_score(estimator=model.classifier, X=X, y=Y.values.ravel(), scoring="f1", cv=k_fold, n_jobs=-1)
# print("{} Folds Mean F1-Score: {}".format(k_fold, f_scores.mean()))

# In[] K-Fold Cross Validation with HappyML
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=model.classifier, k_fold=K, verbose=False)

#print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print(f"{K} Folds Mean Accuracy: {kfp.accuracy():.2%}")
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Visualization
import HappyML.model_drawer as md

md.classify_result(x=X_train, y=Y_train, classifier=model.classifier, title="訓練集 vs. 模型", font='DFKai-sb')
md.classify_result(x=X_test, y=Y_test, classifier=model.classifier, title="測試集 vs. 模型", font='DFKai-sb')
