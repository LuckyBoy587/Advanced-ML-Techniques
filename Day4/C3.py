import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from ML_Modules import split_data

# Suppress all warnings
warnings.filterwarnings('ignore')

def main():
    # Load dataset
    try:
        filename = input().strip()
        filepath = os.path.join(sys.path[0], filename)
        if not os.path.exists(filepath):
            print(f"Error: File '{filename}' not found.")
            sys.exit()
        
        df = pd.read_csv(filepath)
    except Exception:
        sys.exit()

    # Data Preview
    # Set display options to match sample output (6 decimal places for floats)
    pd.options.display.float_format = '{:,.6f}'.format
    print(df.head(5).to_string())
    print()

    # Separate features and target
    X = df[['bmi', 'age', 'insulin', 'FamilyHistory', 'bp']]
    y = df['Fasting blood']

    # Split the dataset (70% training, 30% testing)
    X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

    # Hyperparameter tuning with cross-validation on training set
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    dt_reg = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=dt_reg,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    # Report the best hyperparameters
    best_params = grid_search.best_params_
    # Sort keys alphabetically as per requirement
    sorted_best_params = {k: best_params[k] for k in sorted(best_params)}
    print(f"Best Hyperparameters: {sorted_best_params}")

    # Evaluate model using cross-validation on full dataset
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)
    
    # Format array output to match sample
    print(f"Cross-Validation RMSE Scores: {cv_rmse_scores}")
    print(f"Mean RMSE: {np.mean(cv_rmse_scores)}")

    # Evaluate final model on test data
    # best_model is already refitted on X_train due to refit=True in GridSearchCV
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {test_rmse}")

    # Standard Deviation of Label
    label_std = y.std()
    print(f"Standard Deviation of Label: {label_std}")

    # Interpretation
    if test_rmse <= label_std:
        print("The model's RMSE is within the standard deviation, indicating good performance.")
    else:
        print("The model's RMSE exceeds the standard deviation, suggesting room for improvement.")

if __name__ == "__main__":
    main()
