import sys
import os
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
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
        
        # Check required columns (optional but good practice based on constraints)
        required_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 
                         'average_montly_hours', 'time_spend_company', 'Work_accident', 
                         'left', 'promotion_last_5years', 'Department', 'salary']
        
        # Encode categorical variables
        le = LabelEncoder()
        df['salary.enc'] = le.fit_transform(df['salary'])
        df['Department.enc'] = le.fit_transform(df['Department'])

        # Drop original text columns
        df.drop(['salary', 'Department'], axis=1, inplace=True)

        # Define feature set and target
        X = df.drop('left', axis=1)
        y = df['left']

        # Scale features
        X_scaled = ML_Modules.data_scale(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

        # Hyperparameter Tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
        
        svm = SVC()
        grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Predict on test set using best model
        y_pred = grid.best_estimator_.predict(X_test)

        # Evaluate model
        ML_Modules.evaluate_classifier(y_test, y_pred)

    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
