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


