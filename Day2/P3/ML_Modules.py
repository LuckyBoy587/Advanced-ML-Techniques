import pandas as pd
from sklearn.preprocessing import StandardScaler

def check_correlation(input_df):
    """
    Detects multicollinearity among features.
    Accepts a DataFrame of numeric features (excluding the label).
    Returns a boolean correlation matrix showing True for absolute correlation >= 0.75.
    """
    corr_matrix = input_df.corr()
    # Check for absolute correlation >= 0.75
    bool_corr_matrix = corr_matrix.abs() >= 0.75
    return bool_corr_matrix

def data_scale(input_df):
    """
    Standardizes numeric feature columns.
    Returns a DataFrame of scaled features with original column names preserved.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_df)
    # Convert back to DataFrame to preserve column names
    scaled_df = pd.DataFrame(scaled_data, columns=input_df.columns)
    return scaled_df
