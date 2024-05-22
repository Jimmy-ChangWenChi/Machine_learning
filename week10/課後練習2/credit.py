# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:01:48 2024

@author: jimmy
"""

import HappyML.preprocessor as pp

dataset = pp.dataset(file="CreditCards.csv")

X = pp.decomposition(dataset, x_columns = [i for i in range(18) if i != 0])


X= pp.missing_data(X,strategy="median")

X = pp.feature_scaling(fit_ary=X,transform_arys=X)

#PCA
from HappyML.preprocessor import PCASelector
selector = PCASelector(best_k=2)
X = selector.fit(x_ary= X, verbose = True,plot = True).transform(X)


#K-means
from HappyML.clustering import KMeansCluster

cluster = KMeansCluster()
Y_pred = cluster.fit(x_ary = X,verbose = True,plot=True).predict(x_ary = X,y_column="Credit Type")

# Optional, Attach the Y_pred to Dataset & Save as .CSV file
dataset = pp.combine(dataset, Y_pred)
dataset.to_csv("CreditCards_answers.csv", index=False)

import HappyML.model_drawer as md

md.cluster_drawer(x=X, y=Y_pred, centroids=cluster.centroids, title="Customers Segmentation")



#對加了集群結果做決策樹

dataset = pp.dataset(file="CreditCards_answers.csv")

X,Y = pp.decomposition(dataset, x_columns = [i for i in range(18) if i != 0], y_columns=[18])


# X_columns =['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE'
#             ,'PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY',
#             'CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']


X = pp.missing_data(X,strategy="median")

Y, Y_mapping = pp.label_encoder(Y, mapping=True)

# #Kbest
from HappyML.preprocessor import KBestSelector
selector = KBestSelector()
X = selector.fit(x_ary= X, y_ary=Y,verbose = True,sort= True).transform(X)
X_train, X_test,Y_train,Y_test = pp.split_train_test(x_ary=X, y_ary= Y)


#DecisionTree
from HappyML.classification import DecisionTree
classifier = DecisionTree()
Y_pred = classifier.fit(X_train,Y_train).predict(X_test)



# #十字交叉驗證
from HappyML.performance import KFoldClassificationPerformance
K = 10
kfp = KFoldClassificationPerformance(x_ary=X, y_ary=Y, classifier=classifier.classifier, k_fold=K)

print("----- Random Forest Classification -----") 
print("{} Folds Mean Accuracy: {}".format(K, kfp.accuracy())) 
print("{} Folds Mean Recall: {}".format(K, kfp.recall()))     
print("{} Folds Mean Precision: {}".format(K, kfp.precision())) 
print("{} Folds Mean F1-Score: {}".format(K, kfp.f_score()))   


import HappyML.model_drawer as md
from IPython.display import Image, display

cls_name = str([Y_mapping[key] for key in sorted(Y_mapping.keys())])
#graph = md.tree_drawer(classifier=classifier.classifier, feature_names=X_test.columns, target_names=cls_name, graphviz_bin=GRAPHVIZ_INSTALL)
graph = md.tree_drawer(classifier=classifier.classifier, feature_names=X_test.columns, target_names=cls_name)

display(Image(graph.create_png()))



