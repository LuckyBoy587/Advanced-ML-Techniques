import pandas as pd
import sys
import os
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore')
pd.set_option('display.show_dimensions', False)

def main():
    # Prompt for filename
    try:
        filename = input()
    except EOFError:
        return

    # Construct file path
    file_path = os.path.join(sys.path[0], filename)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{filename}' not found.")
        sys.exit()

    try:
        # Load data
        df = pd.read_csv(file_path)
    except Exception:
        print(f"Error: File '{filename}' is empty or invalid.")
        sys.exit()

    if df.empty:
         print(f"Error: File '{filename}' is empty or invalid.")
         sys.exit()

    # 1. Rows with missing values
    print("Rows with missing values (if any):")
    
    missing_rows = df[df.isnull().any(axis=1)]
    
    if not missing_rows.empty:
        print(missing_rows)
        print(f"\n[{len(missing_rows)} rows x {len(df.columns)} columns]")
        print()
    else:
        print("No missing values found in the dataset.\n")

    # 2. Correlation matrix of numeric columns
    print("Correlation matrix of numeric columns:")
    
    numeric_df = df.select_dtypes(include=['number'])
    
    # Check if we have numeric columns
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        print(corr_matrix)
        print(f"\n[{len(numeric_df.columns)} rows x {len(numeric_df.columns)} columns]")
    else:
        # Fallback if no numeric columns (though constraints say they exist)
        print(pd.DataFrame()) 
        print(f"\n[0 rows x 0 columns]")

if __name__ == "__main__":
    main()
