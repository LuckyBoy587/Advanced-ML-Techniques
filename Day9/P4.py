import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Set pandas options to match sample output truncation
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

def evaluate_clusterer(X, cluster_labels):
    print("========== CLUSTER EVALUATION ==========")
    print("Cluster Evaluation Metrics:")
    s_score = silhouette_score(X, cluster_labels)
    ch_score = calinski_harabasz_score(X, cluster_labels)
    db_score = davies_bouldin_score(X, cluster_labels)

    print(f"Silhouette Score: {s_score:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")

def load_file(filepath):
    if not os.path.exists(filepath):
        # The sample output doesn't show the full path in error, just filename
        filename = os.path.basename(filepath)
        print(f"Error: File '{filename}' not found.")
        sys.exit()
    
    extension = os.path.splitext(filepath)[1].lower()
    if extension == '.csv':
        return pd.read_csv(filepath)
    elif extension in ['.xls', '.xlsx']:
        return pd.read_excel(filepath)
    else:
        print("Error: Unsupported file format. Please use CSV or Excel.")
        sys.exit()

def main():
    print("Enter your dataset filename (CSV or Excel): ")
    # Using the exact instruction for reading dataset
    filepath = os.path.join(sys.path[0], input())
    df = load_file(filepath)
    
    print("========== FIRST 5 ROWS ==========")
    print(df.head())
    print()
    
    print("========== DATASET SHAPE ==========")
    print(df.shape)
    print()
    
    print("========== DATA TYPES ==========")
    print(df.dtypes)
    print()
    
    # Extract numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("Error: No numeric columns found.")
        sys.exit()
        
    print("========== NUMERIC COLUMNS ==========")
    print(list(numeric_df.columns))
    print()
    
    X = numeric_df.values
    
    print("========== SILHOUETTE SCORES ==========")
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"k = {k}: Silhouette Score = {round(score, 3)}")
    print()

    # Final model with k=2
    k_final = 2
    kmeans_final = KMeans(n_clusters=k_final, random_state=10)
    cluster_labels = kmeans_final.fit_predict(X)

    print(f"========== FINAL CLUSTER MODEL (k={k_final}) ==========")
    print("Cluster Labels:")
    print(cluster_labels)
    print()

    evaluate_clusterer(X, cluster_labels)

if __name__ == "__main__":
    main()