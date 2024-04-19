# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:32:50 2024

@author: jimmy
"""

import HappyML.preprocessor as pp


dataset = pp.dataset("Device_Failure.csv")


X, Y = pp.decomposition(dataset, [0],[1])

X_train. X_test,Y_train,Y_test = pp.split_train_test(X, Y, train_size=0.8)


from HappyML.regression import SimpleRegressor
import HappyML.model_drawer as md