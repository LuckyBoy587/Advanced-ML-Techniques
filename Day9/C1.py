import os
import sys
import warnings

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import ML_Modules
from ML_Modules import StandardScaler

# Suppress warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

def main():
    # 1. Dataset Loading
    file_input = input("Enter your data file (CSV or XLSX): ")
    filepath = os.path.join(sys.path[0], file_input)

    if not os.path.exists(filepath):
        print(f"Error: File '{file_input}' not found.")
        return

    # Load dataset based on extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    print("\nDataset Loaded Successfully!")
    print(df.head().to_string())
    print(df.info())

    # 3. Separate input and output
    # Input features: age, children, charges, gender_n, smoker_n, region_n
    # Target variable: weight_condition_n
    features = ['age', 'children', 'charges', 'gender_n', 'smoker_n', 'region_n']
    X = df[features]
    y = df['weight_condition_n']

    print("\nInput Data:")
    print(X.head().to_string())
    print("\nOutput Data:")
    print(y.head().to_string())
    print("Name: weight_condition_n, dtype: int64")
    # 4. Scale the data
    print("\nScaling Input Data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Perform clustering without PCA
    print("\nSilhouette Scores WITHOUT PCA:")
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"k={k}: Silhouette Score = {round(score, 3)}")

    print("\nRunning KMeans WITHOUT PCA...")
    kmeans_no_pca = KMeans(n_clusters=2, random_state=10)
    labels_no_pca = kmeans_no_pca.fit_predict(X_scaled)
    ML_Modules.evaluate_clusterer(X_scaled, labels_no_pca)

    # 6. Apply PCA for dimensionality reduction
    print("\nRunning PCA (n_components=2)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Evaluate silhouette scores with PCA
    print("\nSilhouette Scores WITH PCA:")
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=10)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"k={k}: Silhouette Score = {round(score, 3)}")

    print("\nRunning KMeans WITH PCA...")
    kmeans_pca = KMeans(n_clusters=2, random_state=10)
    labels_pca = kmeans_pca.fit_predict(X_pca)
    ML_Modules.evaluate_clusterer(X_pca, labels_pca)

    # 11. Final Summary
    print("\n" + "="*20 + " SUMMARY " + "="*20)
    print("Data Loaded")
    print("Data Scaled")
    print("Optimal k checked using Silhouette Score")
    print("K-Means applied WITHOUT PCA")
    print("K-Means applied WITH PCA")
    print("Evaluation completed using silhouette score")
    print("="*50)

if __name__ == "__main__":
    main()
