# Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set and visualize the clustering result.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce the dataset to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform K Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.scatter(kmeans.cluster_centers_[:,0], 
    kmeans.cluster_centers_[:, 1],
    c="red", s=200, marker='x', label='Centroids'
)
plt.title("K-Means Clustering on Wisconsin Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()



# Evaluate Clustering by comparing with true labels
from sklearn.metrics import adjusted_rand_score
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y, y_kmeans):.2f}")