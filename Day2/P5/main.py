import sys
import os
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import ML_Modules

# Suppress warnings
warnings.simplefilter(action='ignore')

def main():
    try:
        # Input CSV file name
        file_name = input()
        file_path = os.path.join(sys.path[0], file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_name}' not found.")
            return

        # Load dataset
        df = pd.read_csv(file_path)

        # Check if target column exists
        if 'left' not in df.columns:
            print("Error: 'left' column missing.")
            return

        # Encode categorical variables
        le = LabelEncoder()
        df['salary.enc'] = le.fit_transform(df['salary'])
        df['Department.enc'] = le.fit_transform(df['Department'])

        # Drop original text columns
        df.drop(['salary', 'Department'], axis=1, inplace=True)

        # Define feature set and target
        # Feature set after encoding: satisfaction_level, last_evaluation, number_project, 
        # average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, 
        # salary.enc, Department.enc
        
        feature_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 
                        'average_montly_hours', 'time_spend_company', 'Work_accident', 
                        'promotion_last_5years', 'salary.enc', 'Department.enc']
        
        X = df[feature_cols]
        y = df['left']

        # Scale features
        X_scaled = ML_Modules.custom_scaling(X)

        # Split data (Sequential split)
        X_train, X_test, y_train, y_test = ML_Modules.custom_train_test_split(X_scaled, y, train_ratio=0.8)

        # Train SVM model
        svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        svm.fit(X_train, y_train)

        # Predict on test set
        y_pred = svm.predict(X_test)

        # Evaluate model
        ML_Modules.evaluate_classification(y_test, y_pred)

    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
