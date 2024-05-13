# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:20:09 2019

@author: 俊男
"""

# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="Mall_Customers.csv")

# Decomposition
X = pp.decomposition(dataset, x_columns=[1, 2, 3, 4])

# One-Hot Encoding
X = pp.onehot_encoder(ary=X, columns=[0], remove_trap=True)

# Feature Scaling (for PCA Feature Selection)
X = pp.feature_scaling(fit_ary=X, transform_arys=X)

# Feature Selection (PCA)
from HappyML.preprocessor import PCASelector

selector = PCASelector()
X = selector.fit(x_ary=X, verbose=True, plot=True).transform(x_ary=X)

# In[] K-Means Clustering with Fixed Clusters = 4 (Without HappyML)
# from sklearn.cluster import KMeans
# import time

# # K-Means Clustering with K=4
# kmeans = KMeans(n_clusters=4, init="k-means++", random_state=int(time.time()))
# Y_pred = kmeans.fit_predict(X)

# In[] K-Means Clustering with Visual Clusters, which is also = 4 (Without HappyML)
# from sklearn.cluster import KMeans
# import time

# # Find Best K
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init="k-means++", random_state=int(time.time()))
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # # Draw WCSS for each K
# import matplotlib.pyplot as plt

# plt.plot(range(1, 11), wcss)
# plt.title("The Best K")
# plt.xlabel("# of Clusters")
# plt.ylabel("WCSS")
# plt.show()

# # Clustering with Visual K, which is also =4
# kmeans = KMeans(n_clusters=4, init="k-means++", random_state=int(time.time()))
# Y_pred = kmeans.fit_predict(X)

# In[] K-Means Clustering (With HappyML's Class)
from HappyML.clustering import KMeansCluster

cluster = KMeansCluster()
Y_pred = cluster.fit(x_ary=X, verbose=True, plot=True).predict(x_ary=X, y_column="Customer Type")

# Optional, Attach the Y_pred to Dataset & Save as .CSV file
dataset = pp.combine(dataset, Y_pred)
dataset.to_csv("Mall_Customers_answers.csv", index=False)

# In[] Visualization (Without HappyML's Class)
# import matplotlib.pyplot as plt

# # Draw Samples
# Y_array = Y_pred.values.ravel()
# plt.scatter(X.iloc[Y_array==0, 0], X.iloc[Y_array==0, 1], s=50, c="red", label="Cluster 0")
# plt.scatter(X.iloc[Y_array==1, 0], X.iloc[Y_array==1, 1], s=50, c="blue", label="Cluster 1")
# plt.scatter(X.iloc[Y_array==2, 0], X.iloc[Y_array==2, 1], s=50, c="green", label="Cluster 2")
# plt.scatter(X.iloc[Y_array==3, 0], X.iloc[Y_array==3, 1], s=50, c="cyan", label="Cluster 3")

# # Draw Centroids
# centroids = cluster.centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="black", marker="^", label="Centroids")

# # Labels & Legends
# plt.title("Clusters of Clients")
# plt.xlabel("PCA1")
# plt.ylabel("PCA2")
# plt.legend(loc="best")
# plt.show()

# In[] Visualization (With HappyML's Class)
import HappyML.model_drawer as md

md.cluster_drawer(x=X, y=Y_pred, centroids=cluster.centroids, title="Customers Segmentation")