import pandas as pd
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_scale(X_DT):
    # Select only the numeric columns as per constraints
    numeric_cols = X_DT.select_dtypes(include=['number']).columns
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(X_DT[numeric_cols])
    
    # Return as a DataFrame with original column names
    return pd.DataFrame(scaled_data, columns=numeric_cols)

def main():
    # Get filename from user
    filename = input().strip()
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    # 1. Load Data
    df = pd.read_csv(filename)
    print(f"The number of samples in data is {len(df)}.")

    # 2. Data Types
    print("\nData Types:")
    print(df.dtypes)

    # 3. Numeric Summary
    print("\nNumeric Summary:")
    summary = df.describe()
    print(summary)
    print(f"\n[{summary.shape[0]} rows x {summary.shape[1]} columns]")

    # 4. Drop Irrelevant Columns
    # Removing original categorical text columns as per requirements
    df_dropped = df.drop(columns=['Department', 'salary'])
    print("\nData After Dropping Irrelevant Columns:")
    df_dropped.info()

    # 5. Split Features and Target
    # Target is 'average_monthly_hours'
    input_df = df_dropped.drop(columns=['average_monthly_hours'])
    output_df = df_dropped['average_monthly_hours']

    print("\nInput Features:")
    print(input_df.head())
    print(f"\n[{input_df.head().shape[0]} rows x {input_df.shape[1]} columns]")

    # 6. Target Variable
    print("\nTarget Variable:")
    print(output_df.head())

    # 7. Scaled Feature Data
    scaled_df = data_scale(input_df)
    print("\nScaled Feature Data:")
    print(scaled_df.head())
    print(f"\n[{scaled_df.head().shape[0]} rows x {scaled_df.shape[1]} columns]")

if __name__ == "__main__":
    main()