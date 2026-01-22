import os
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv(os.path.join(sys.path[0], input()))
print("Enter your dataset filename (CSV): First 5 rows of the dataset:")

print(df.head())
print("Number of samples and features:")
print(df.shape)

print("Data types of each column:")
print(df.dtypes)
df = df.select_dtypes(include=["number"])
print("Numeric columns used for clustering:")
print(df.columns.tolist())

print("Original shape:", df.shape)

pca = PCA(n_components=2)
pca_values = pca.fit_transform(df)
pca_df = pd.DataFrame(pca_values)

print("Reduced shape after PCA:", pca_df.shape)
print("Silhouette Scores for different cluster sizes:")

for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    labels = kmeans.fit_predict(pca_values)
    sil_score = silhouette_score(pca_values, labels)
    print(f"For n_clusters = {n_clusters}, Average Silhouette Score = {round(sil_score, 3)}")

print("Cluster labels assigned (K=3):")

kmeans = KMeans(n_clusters=3, random_state=10)
labels = kmeans.fit_predict(pca_values)
print(labels)

print("========== Cluster Evaluation Metrics ==========")
print(f"Silhouette Score: {silhouette_score(pca_values, labels):.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(pca_values, labels):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(pca_values, labels):.4f}")