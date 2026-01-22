import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import ML_Modules
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Dataset Loading
    # Using the path joining logic as per instructions
    file_input = input("Enter the housing data file (e.g., ML470_S9_KCHouse_Data_Practice.xlsx): ")
    filepath = os.path.join(sys.path[0], file_input)

    if not os.path.exists(filepath):
        print(f"Error: File '{file_input}' not found.")
        return

    df = pd.read_excel(filepath)
    print("\nDataset Loaded Successfully!")
    print(f"Initial columns: {df.columns.tolist()}")

    # 2. Data Preparation (Based on P1.ipynb)
    # Drop unimportant variables
    df.drop(['id', 'date'], axis=1, inplace=True)

    # Transform and categorize house prices into quartiles
    df['price_range'] = pd.qcut(df['price'], q=4, labels=['Low', 'Mid', 'Upper-Mid', 'High'])
    
    # Encode the labels
    le = LabelEncoder()
    df['price_range_encoded'] = le.fit_transform(df['price_range'])

    # Treat outliers in numerical features
    # Note: We drop categorical 'price_range' before treating outliers
    df_for_outliers = df.drop(['price_range'], axis=1)
    df_treated = ML_Modules.treat_outliers(df_for_outliers)

    # Resolve multicollinearity (Based on P1.ipynb)
    # Dropping sqft_living and sqft_lot due to high correlation with other features
    df_selected = df_treated.drop(['sqft_living', 'sqft_lot', 'price'], axis=1)
    
    print("\nPreprocessed Data Info:")
    print(df_selected.info())
    print("\nFeatures used for classification:", df_selected.drop('price_range_encoded', axis=1).columns.tolist())

    # 3. Separate Features and Target
    X = df_selected.drop('price_range_encoded', axis=1)
    y = df_selected['price_range_encoded']

    # 4. Scale the Data
    print("\nScaling Features...")
    X_scaled = ML_Modules.data_scale(X)

    # 5. Split Dataset (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, shuffle=True)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # 6. Identify Optimum K Value
    print("\nFinding optimal K value (1 to 40)...")
    error_rate = []
    for i in range(1, 41):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    # Select K with minimum error rate
    best_k = error_rate.index(min(error_rate)) + 1
    print(f"Optimal K Value identified: {best_k}")

    # Plot Error Rate vs K
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 41), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.grid(True)
    # In a script, plt.show() might block or be ignored depending on environment, 
    # but it's good practice to include it.
    plt.show()

    # 7. Build Final KNN Classifier
    print(f"\nBuilding final KNN classifier with K={best_k}...")
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)

    # 8. Make Predictions
    y_pred = knn_final.predict(X_test)
    y_prob = knn_final.predict_proba(X_test)

    # 9. Evaluate Model Performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

    print("\n" + "="*30)
    print(" KNN CLASSIFICATION RESULTS ")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
