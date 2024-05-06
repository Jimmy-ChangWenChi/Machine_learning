# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 00:07:56 2019

@author: 俊男
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Data
dataset = pp.dataset(file="Match_Making.csv")

# Decomposition
X, Y = pp.decomposition(dataset, x_columns=[i for i in range(1, 5)], y_columns=[5])

# Dummy Variables
X = pp.onehot_encoder(X, columns=[3], remove_trap=True)
Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# Feature Selection
#from HappyML.preprocessor import KBestSelector
#selector = KBestSelector(best_k="auto")
#X = selector.fit(x_ary=X, y_ary=Y, verbose=True, sort=True).transform(x_ary=X)

# Split Training / Testing Set
#X_train, X_test, Y_train, Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

# In[] Decision Tree
from HappyML.classification import DecisionTree

classifier = DecisionTree()
Y_pred = classifier.fit(X, Y).predict(X)

# In[] Performance
from HappyML.performance import KFoldClassificationPerformance

K = 3
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Decision Tree Classification -----")
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy()))
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))
print("{} Folds Mean Precision: {}".format(K, kfp.precision()))
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))

# In[] Visualization
import HappyML.model_drawer as md
from IPython.display import Image

cls_name = [Y_mapping[key] for key in sorted(Y_mapping.keys())]
graph = md.tree_drawer(classifier=classifier.classifier, feature_names=X.columns, target_names=cls_name, graphviz_bin='C:/Program Files (x86)/Graphviz2.38/bin/')
Image(graph.create_png())
