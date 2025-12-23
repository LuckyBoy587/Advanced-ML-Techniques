import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sys
import os
import warnings
import ML_Modules

# Suppress warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

def main():
    try:
        filename = input()
        file_path = os.path.join(sys.path[0], filename)

        if not os.path.exists(file_path):
            print(f"Error: File '{filename}' not found.")
            return

        df = pd.read_csv(file_path)

        # 1. First 5 rows
        print(df.head())
        print()

        # 2. Data types
        print(df.dtypes)
        print()

        # 3. Features and Target
        features = ['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']
        target = 'Outcome'
        
        X = df[features]
        y = df[target]

        # 4. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # 5. Train
        model = GaussianNB()
        model.fit(X_train, y_train)
        print("Model trained.")
        print()

        # 6. Predict
        y_pred = model.predict(X_test)
        print(f"Predicted Values: {repr(y_pred)}")
        print()

        # 7. Evaluate
        ML_Modules.evaluate_classifier(y_test, y_pred)

    except Exception as e:
        # Fallback for unforeseen errors, though logic above handles the main file check
        # If the file path construction fails or something else
        if "No such file" in str(e):
             # This block might not be reached due to explicit check, but good for safety
             pass
        else:
             raise e

if __name__ == "__main__":
    main()
