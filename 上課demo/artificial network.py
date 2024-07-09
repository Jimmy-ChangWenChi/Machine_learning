# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:21:05 2024

@author: jimmy
"""

import HappyML.preprocessor as pp

dataset = pp.dataset("Mushroom.csv")

X,Y = pp.decomposition(dataset, x_columns=[i for i in range(1,23)], y_columns=[0])

x = pp.onehot_encoder(X,columns=[i for i in range(22)],remove_trap = True)


Y, Y_mapping = pp.label_encoder(Y,mapping=True)


from HappyML.preprocessor import KBestSelector

selector = KBestSelector()

X = selector.fit(X,Y,verbose = True,sort = True).transform(X)

X_train,X_test,Y_train,Y_test = pp.split_train_test(X, Y)

