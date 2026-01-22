from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

def evaluate_clusterer(X, labels):
    """
    Computes and displays silhouette score and cluster member counts.
    """
    score = silhouette_score(X, labels)
    print("\nCluster Evaluation:")
    print(f"Silhouette Score: {score:.3f}")
    print("\nCluster Member Counts:")
    print(pd.Series(labels).value_counts())

def assess_outliers(data1):
    """
    Detects existence of outliers in numerical features using IQR method.
    Returns a dictionary indicating if outliers exist for each column.
    """
    outlier_status = {}
    for col in data1.select_dtypes(include=[np.number]).columns:
        Q1 = data1[col].quantile(0.25)
        Q3 = data1[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        has_outliers = ((data1[col] < lower_bound) | (data1[col] > upper_bound)).any()
        outlier_status[col] = has_outliers
    return outlier_status

def treat_outliers(data1):
    """
    Impute outliers using IQR winsorization method.
    Caps extreme values at calculated thresholds.
    """
    data_treated = data1.copy()
    for col in data_treated.select_dtypes(include=[np.number]).columns:
        Q1 = data_treated[col].quantile(0.25)
        Q3 = data_treated[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Winsorization: Cap values
        data_treated[col] = np.where(data_treated[col] < lower_bound, lower_bound, data_treated[col])
        data_treated[col] = np.where(data_treated[col] > upper_bound, upper_bound, data_treated[col])
        
    return data_treated

def data_scale(X_DT):
    """
    Standardize features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_DT)
    return X_scaled
