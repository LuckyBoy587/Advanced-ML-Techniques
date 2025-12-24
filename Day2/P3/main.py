import pandas as pd
import os
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ML_Modules as mm

# Suppress warnings
warnings.filterwarnings("ignore")
pd.set_option('display.show_dimensions', False)

def main():
    try:
        # Prompt user for CSV file name
        file_name = input()
        
        # Construct file path using sys.path[0] as requested in GEMINI.md context
        # However, for robustness in this specific script run, I'll use the current directory logic 
        # or just the filename if it's expected to be in the same dir. 
        # The prompt says "The program prompts the user to enter the name of the CSV file... Input must include the file extension .csv."
        # The context says "use os.path.join(sys.path[0], input())".
        
        file_path = os.path.join(sys.path[0], file_name)

        if not os.path.exists(file_path):
             print(f"Error: File '{file_name}' not found.")
             return

        df = pd.read_csv(file_path)

        # Validate columns
        required_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 
                            'average_montly_hours', 'time_spend_company', 'Work_accident', 
                            'left', 'promotion_last_5years', 'Department', 'salary']
        
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain exactly these columns: {required_columns}")
            return
            
        if 'left' not in df.columns:
            print("Error: 'left' column is missing.")
            return

        # 1. Label Encoding Confirmation
        print("=== Label Encoding Categorical Columns ===")
        
        le_salary = LabelEncoder()
        df['salary.enc'] = le_salary.fit_transform(df['salary'])
        encoded_salary_classes = sorted(le_salary.classes_.tolist())
        print(f"Encoded salary classes: {encoded_salary_classes}")
        
        le_dept = LabelEncoder()
        df['Department.enc'] = le_dept.fit_transform(df['Department'])
        encoded_dept_classes = sorted(le_dept.classes_.tolist())
        print(f"Encoded Department classes: {encoded_dept_classes}")
        
        # Drop original categorical columns
        df_encoded = df.drop(['salary', 'Department'], axis=1)

        # 2. Features and Label Separation
        print("\n=== Separating Features and Label ===")
        y = df_encoded['left']
        X = df_encoded.drop('left', axis=1)
        
        print(f"Input Features Shape: {X.shape}")
        print(f"Label Shape: {y.shape}")

        # 3. Correlation Boolean Matrix
        print("\n=== Correlation Boolean Matrix (correlation >= 0.75) ===")
        corr_bool = mm.check_correlation(X)
        print(corr_bool)
        print(f"\n[{corr_bool.shape[0]} rows x {corr_bool.shape[1]} columns]")

        # 4. Scaled Feature Sample
        print("\n=== Scaled Feature Sample (First 5 Rows) ===")
        X_scaled = mm.data_scale(X)
        print(X_scaled.head(5))
        print(f"\n[5 rows x {X.shape[1]} columns]")

        # 5. Train-Test Split Information
        print("\n=== Splitting Data into Train (80%) and Test (20%) ===")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)
        
        print(f"Training Features Shape: {X_train.shape}")
        print(f"Training Labels Shape: {y_train.shape}")
        print(f"Testing Features Shape: {X_test.shape}")
        print(f"Testing Labels Shape: {y_test.shape}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
