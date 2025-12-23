import pandas as pd
import os
import sys

def main():
    try:
        # Prevent pandas from printing dimensions automatically, so we can control it manually
        pd.set_option('display.show_dimensions', False)

        filename = input()
        file_path = os.path.join(sys.path[0], filename)
        
        if not os.path.exists(file_path):
            print(f"Error: File '{filename}' not found.")
            return

        df = pd.read_csv(file_path)

        # 1. First 5 rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print(f"\n[{df.head().shape[0]} rows x {df.shape[1]} columns]")

        # 2. Number of samples
        print("\nNumber of samples in the data:")
        print(df.shape[0])

        # 3. Data types
        print("\nData types of each column:")
        print(df.dtypes)

        # 4. Feature columns
        feature_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
        print("\nFeature columns used for classification:")
        print(feature_cols)

        # 5. Statistical summary
        print("\nStatistical summary of numeric columns:")
        # describe() by default handles numeric types.
        # Department and salary are objects, left is int64 (numeric).
        # The prompt says: "excludes Department and salary". 
        # By default describe() includes all numeric columns. 'left' is int64 so it will be included.
        summary = df.describe()
        print(summary)
        print(f"\n[{summary.shape[0]} rows x {summary.shape[1]} columns]")

        # 6. Sample categorical data
        cat_cols = ['Department', 'salary', 'left']
        missing_cols = [col for col in cat_cols if col not in df.columns]
        
        if missing_cols:
             print("Categorical columns ('Department', 'salary', 'left') not found in dataset.")
        else:
            print("\nSample categorical data (Department, salary, left):")
            print(df[cat_cols].head())

    except Exception as e:
        # In case of other errors, though prompt specifies just file not found.
        # But for safety, we might just let python print trace or handle gracefully?
        # The prompt only specified "Error: File '{filename}' not found." handling.
        # I will assume standard execution otherwise.
        # If input() fails or something else, default behavior is fine.
        raise e

if __name__ == "__main__":
    main()
