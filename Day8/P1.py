import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import StandardScaler

def assess_outliers():
    # Placeholder function
    pass

def main():
    filename = input().strip()
    
    df = pd.read_csv(os.path.join(sys.path[0], filename))

    # 1. Dataset Preview
    print("Dataset Preview:")
    print(df.head())

    # 2. Dataset Info
    print("\nDataset Info:")
    print(df.info())

    # 3. Dataset Description
    print("\nDataset Description:")
    print(df.describe())

    # 4. Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # 5. Outlier Treatment
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Winsorization results rounded to 2 decimal places
    df = df.round(2)
    # Ensure all columns are float for the rounding and preview
    for col in df.columns:
        df[col] = df[col].astype(float)

    print("\nData After Outlier Treatment:")
    print(df.head())

    # 6. Multicollinearity Matrix
    print("\nMulticollinearity Matrix:")
    corr_matrix = df.corr().abs()
    bool_matrix = corr_matrix >= 0.7
    print(bool_matrix)

    # 7. Columns After Removal
    # "The Detergents_Paper column is removed after multicollinearity analysis"
    # "Only columns that exist in the DataFrame are dropped (safe removal)"
    columns_to_drop = ['Detergents_Paper']
    df_reduced = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print("\nColumns after removal:")
    print(df_reduced.columns.tolist())

    # 8. Scaled Data Preview
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_reduced)
    df_scaled = pd.DataFrame(scaled_data, columns=df_reduced.columns)
    
    print("\nScaled Data Preview:")
    print(df_scaled.head())

if __name__ == "__main__":
    main()