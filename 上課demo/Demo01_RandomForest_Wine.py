    # -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:57:58 2019

@author: 俊男
"""

# In[] Preprocessing #1: Without PCA and Boundary Visualization
import HappyML.preprocessor as pp

# Load Data, also can be loaded by sklearn.datasets.load_wine()
dataset = pp.dataset(file="Wine.csv")

# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(13)], y_columns=[13])

# By KBestSelector
from HappyML.preprocessor import KBestSelector
selector = KBestSelector(best_k="auto")
X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# In[] Preprocessing #2: Without PCA, but with Boundary Visualization
# import HappyML.preprocessor as pp

# # Load Data, also can be loaded by sklearn.datasets.load_wine()
# dataset = pp.dataset(file="Wine.csv")

# # Decomposition
# X, Y = pp.decomposition(dataset, x_columns=[i for i in range(13)], y_columns=[13])

# # By KBestSelector
# from HappyML.preprocessor import KBestSelector
# selector = KBestSelector(best_k=2)
# X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# # Split Training / Testing Set
# X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# # Feature Scaling
# X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))

# In[] Preprocessing #3: With PCA, and Boundary Visualization
import HappyML.preprocessor as pp

# Load Data, also can be loaded by sklearn.datasets.load_wine()
dataset = pp.dataset(file="Wine.csv")

# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(13)], y_columns=[13])

# Feature Scaling
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# # PCA without HappyML's Class
# from sklearn.decomposition import PCA
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# pca = PCA(n_components=None)
# pca.fit(X)
# info_covered = pca.explained_variance_ratio_
# cumulated_sum = np.cumsum(info_covered)
# plt.plot(cumulated_sum, color="blue")

# pca = PCA(n_components=2)
# X_columns = ["PCA_{}".format(i+1) for i in range(2)]
# X = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=X_columns)

# PCA with HappyML's Class
from HappyML.preprocessor import PCASelector

selector = PCASelector(best_k="auto")
X = selector.fit(x_ary=X, verbose=True, plot=True).transform(X)

# Split Training / Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# In[] Random Forest

# # Without HappyML's Class
# from sklearn.ensemble import RandomForestClassifier
# import time

# classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=int(time.time()))
# classifier.fit(X_train, Y_train.values.ravel())
# Y_pred = classifier.predict(X_test)

# With HappyML's Class   
from HappyML.classification import RandomForest

classifier = RandomForest(n_estimators=10, criterion="entropy")
Y_pred = classifier.fit(X_train, Y_train).predict(X_test)

# In[] Performance
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Random Forest Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Tree Visualization
#GRAPHVIZ_INSTALL = "C:/Program Files (x86)/Graphviz/bin"

import HappyML.model_drawer as md
from IPython.display import Image, display

#for i in range(10):
    clfr = classifier.classifier.estimators_[1]
    #graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names="123", graphviz_bin=GRAPHVIZ_INSTALL)
    #graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names="123")
    graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names=["1","2","3"])

    display(Image(graph.create_png()))

# In[] Boundary Visualization
# import HappyML.model_drawer as md

# md.classify_result(x=X_train, y=Y_train, classifier=classifier.classifier,
#                    fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"),
#                    title="訓練集 vs. 隨機森林模型", font="DFKai-sb")
# md.classify_result(x=X_test, y=Y_test, classifier=classifier.classifier,
#                    fg_color=("orange", "blue", "white"), bg_color=("red", "green", "black"),
#                    title="測試集 vs. 隨機森林模型", font="DFKai-sb")
