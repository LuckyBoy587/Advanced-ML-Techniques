import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
def solve():
    filename = input().strip()
    try:
        path = os.path.join(sys.path[0], filename)
        df = pd.read_csv(path)
    except Exception:
        print(f"Error: Unable to read file '{filename}'.")
        return
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    print("--- Outlier Assessment ---")
    numeric_cols = df.columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers")
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    if 'Purchase Likelihood' not in df.columns:
        print("Error: Target column 'Purchase Likelihood' not found in dataset.")
        return
    X = df.drop('Purchase Likelihood', axis=1)
    y = df['Purchase Likelihood']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("\n==============================")
    print(f"Model Accuracy: {round(accuracy, 2)} %")
    print("==============================")
if __name__ == "__main__":
    solve()