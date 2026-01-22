import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Suppress all warnings
warnings.filterwarnings("ignore")

def main():
    # Prompt for the CSV file name
    filename = input("Enter the name of the CSV file: ")
    filepath = os.path.join(sys.path[0], filename)

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File '{filename}' not found.")
        return

    try:
        # Load the dataset
        df = pd.read_csv(filepath)

        # Extract features and target variable
        # Columns: Fasting blood, bmi, age, FamilyHistory, HbA1c, target
        feature_cols = ['Fasting blood', 'bmi', 'age', 'FamilyHistory', 'HbA1c']
        X = df[feature_cols].values
        y = df['target'].values.ravel()

        # Configure Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define the hyperparameter search grid
        param_grid = {
            'max_depth': [2, 3, 4, 5, 6],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 3]
        }

        # Initialize DecisionTreeClassifier
        dtc = DecisionTreeClassifier(random_state=42)

        # Execute GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=dtc,
            param_grid=param_grid,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)

        # Identify the best-performing model configuration
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Report optimization results
        # Keys in dictionary are ordered alphabetically automatically in Python 3.7+ if we define them correctly
        # or we can just print the dictionary.
        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Stratified CV Accuracy: {best_score:.3f}")

    except Exception as e:
        # Generic error handling if needed, though instructions are specific about file not found
        pass

if __name__ == "__main__":
    main()
