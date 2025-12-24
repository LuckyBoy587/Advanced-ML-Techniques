import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder

def main():
    try:
        filename = input().strip()
    except EOFError:
        return

    # Context requires os.path.join(sys.path[0], input())
    # But usually sys.path[0] is the script directory.
    file_path = os.path.join(sys.path[0], filename)

    if not os.path.exists(file_path):
        print(f"Error: File '{filename}' not found.")
        return

    # Disable pandas dimension printing to avoid duplication
    pd.set_option('display.show_dimensions', False)

    df = pd.read_csv(file_path)

    # 1. First 5 Rows of Data
    print("=== First 5 Rows of Data ===")
    print(df.head())
    print(f"\n[{len(df.head())} rows x {len(df.columns)} columns]")

    # 2. Number of Samples
    print(f"\nThe number of samples in data is {len(df)}.")

    # 3. Data Types
    print("\n=== Data Types ===")
    print(df.dtypes)

    # 4. Statistical Summary
    print("\n=== Statistical Summary (Describe) ===")
    desc = df.describe()
    print(desc)
    print(f"\n[{len(desc)} rows x {len(desc.columns)} columns]")

    # 5. Missing Values Per Column
    print("\n=== Missing Values Per Column ===")
    print(df.isnull().sum())

    # 6. Salary Encoding Classes
    print("\n=== Salary Encoding Classes ===")
    unique_salary = sorted(df['salary'].unique().tolist())
    print(unique_salary)
    
    le_salary = LabelEncoder()
    le_salary.fit(unique_salary)
    df['salary.enc'] = le_salary.transform(df['salary'])

    # 7. Department Encoding Classes
    print("\n=== Department Encoding Classes ===")
    unique_dept = sorted(df['Department'].unique().tolist())
    print(unique_dept)

    le_dept = LabelEncoder()
    le_dept.fit(unique_dept)
    df['Department.enc'] = le_dept.transform(df['Department'])

    # 8. Dropping Columns
    print("\n=== Dropping 'Department' and 'salary' columns ===")
    df.drop(columns=['Department', 'salary'], inplace=True)

    # 9. Updated DataFrame Info
    print("\n=== Updated DataFrame Info ===")
    df.info()

if __name__ == "__main__":
    main()
