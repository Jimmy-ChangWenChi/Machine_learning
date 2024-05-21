# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:01:36 2024

@author: jimmy
"""

import HappyML.preprocessor as pp


dataset = pp.dataset(file="Zoo_Data.csv")

X, Y = pp.decomposition(dataset, x_columns=[i for i in range(16) if i != 3], y_columns=[17])

X = pp.onehot_encoder(X, columns = [0], remove_trap = True)

target_names = ["Mammal","Bird","Reptile","Fish","Amphibian","Bug","Invertebrate"]
dataset_className = pp.dataset(file="Zoo_Class_Name.csv")

class_names = [row["Class_Type"] for index, row in dataset_className.iterrows()]

#KBest
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(X, Y,verbose = True,sort= True).transform(X)
X_train,X_test,Y_train,Y_test = pp.split_train_test(x_ary=X, y_ary=Y)

#PCA

#Feature Scaling
# X = pp.feature_scaling(fit_ary=X,transform_arys=X)

# from HappyML.preprocessor import PCASelector
# selector = PCASelector(best_k='auto')
# X = selector.fit(X,verbose = True,plot= True).transform(X)

# X_train, X_test,Y_train,Y_test = pp.split_train_test(X, Y)



#RandomForest
from HappyML.classification import RandomForest
classifier = RandomForest(n_estimators=10, criterion="entropy")
Y_pred = classifier.fit(X_train,Y_train).predict(X_test)



#十字交叉驗證
from HappyML.performance import KFoldClassificationPerformance

K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Random Forest Classification -----") 
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy())) #KBEST =0.95 , PCA = 0.93
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))     #KBEST = 0.892, PCA = 0.88
print("{} Folds Mean Precision: {}".format(K, kfp.precision())) #KBEST = 0.87, PCA = 0.83
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))    #KBEST = 0.878, PCA = 0.849


import HappyML.model_drawer as md
from IPython.display import Image, display

for i in range(10):
    clfr = classifier.classifier.estimators_[i]
    #graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names="123", graphviz_bin=GRAPHVIZ_INSTALL)
    #graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names="123")
    graph = md.tree_drawer(classifier=clfr, feature_names=X_test.columns, target_names=class_names)

    display(Image(graph.create_png()))
