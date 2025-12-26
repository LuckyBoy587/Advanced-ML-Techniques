import pandas as pd
import sys
import os
import ML_Modules
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import numpy as np

def main():
    # Input file path
    try:
        filename = input()
        file_path = os.path.join(sys.path[0], filename)
        
        if not os.path.exists(file_path):
             print(f"Error: File '{filename}' not found.")
             sys.exit(0)
             
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: File '{filename}' not found.")
        sys.exit(0)

    # 1. Head of All Columns
    print("# Head of all columns")
    print(df.head())

    # 2. Data Types of All Columns
    print("# Data Types of all columns")
    print(df.dtypes)
    print()

    # Define working columns
    working_cols = ['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c', 'Outcome']
    df_subset = df[working_cols].copy()

    # 3. Working Subset Head
    print("# Working subset head")
    print(df_subset.head().to_string())
    print()

    # 4. Mean Values Grouped by Outcome
    print("# Mean values grouped by Outcome")
    print(df_subset.groupby('Outcome').mean().to_string())
    print()

    # 5. Null Value Check
    print("# Null value check")
    print(df_subset.isnull().sum())
    print()

    # 6. Zero-Value Counts (Before Removal)
    print("# Zero-value count for BMI")
    print(df_subset[df_subset['BMI'] == 0].shape[0])
    print()

    print("# Zero-value count for Glucose")
    print(df_subset[df_subset['Glucose'] == 0].shape[0])
    print()
    
    print("# Zero-value count for Age")
    print(df_subset[df_subset['Age'] == 0].shape[0])
    print()

    # Remove zero values
    # "Remove records with invalid zero values in essential clinical measurements" -> Glucose, BMI, Age
    df_clean = df_subset[(df_subset['Glucose'] != 0) & (df_subset['BMI'] != 0) & (df_subset['Age'] != 0)].copy()

    # 7. Zero-Value Counts (After Removal)
    print("# Zero-value count after removal: Glucose")
    print(df_clean[df_clean['Glucose'] == 0].shape[0])
    print()

    print("# Zero-value count after removal: BMI")
    print(df_clean[df_clean['BMI'] == 0].shape[0])
    print()

    # 8. Row Count After Zero Removal
    print("# Number of rows after zero-value removal")
    print(df_clean.shape[0])
    print()

    # 9. Data After Outlier Treatment
    # Apply to feature columns only (exclude Outcome)
    features_to_treat = ['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']
    
    # Use ML_Modules.treat_outliers. 
    # Note: ML_Modules.treat_outliers treats all numeric columns.
    # We should pass the features part, treat them, then recombine with Outcome.
    
    X_temp = df_clean[features_to_treat]
    X_treated = ML_Modules.treat_outliers(X_temp)
    
    df_final = X_treated.copy()
    df_final['Outcome'] = df_clean['Outcome']
    
    print("# Data after outlier treatment")
    # Output format shows floats with one decimal place.
    # We can format the string output.
    # The sample output has indices 0, 1, 2, 3, 4, 5, 6, 7, 8, 10 ... (index 9 removed?)
    # Sample index 9 in sample data:
    # 125,96,0,0,0,0.232,54,1,11,1 -> BMI is 0. So it was removed.
    # The indices in sample output are not reset.
    
    # To match the output format exactly (aligned columns), to_string is good but we need decimal formatting.
    # pd.options.display.float_format = '{:.1f}'.format affects global.
    # Or apply format to the print.
    
    # Check sample output for float formatting.
    # Glucose 148.0, BMI 33.6, FamilyHistory 0.0
    # integers are also printed as floats ending in .0
    
    # We can convert features to float for printing purposes
    print_df = df_final.drop(columns=['Outcome']).copy()
    for col in features_to_treat:
        print_df[col] = print_df[col].astype(float)
    
    int_cols = ["Age", "HbA1c"]
    for col in int_cols:
        print_df[col] = print_df[col].astype(int)
    # Formatting to 1 decimal place
    # We can use map inside to_string? or formatting.
    # Let's try general format first.
    print(print_df.to_string(float_format='%.1f'))
    print()

    # 5. Model Development
    X = df_final[features_to_treat]
    y = df_final['Outcome']

    # Scale
    X_scaled = ML_Modules.data_scale(X)

    # Split
    X_train, X_test, y_train, y_test = ML_Modules.split_data(X_scaled, y, 0.2)

    # Train SVM
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict
    y_pred = svm_model.predict(X_test)

    # 10. SVM Model Evaluation
    print("# SVM Model Evaluation")
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("===================")
    print()

    print("Classification Report:")
    # The sample output for classification report has specific spacing.
    # standard classification_report returns a string.
    print(classification_report(y_test, y_pred))
    print("===================")

    # Individual Metrics
    # Sample:
    # accuracy: 0.600
    # recall: 0.800
    # f1-score: 0.667
    # precision: 0.571
    
    # Note: These are likely for the positive class (1) or overall accuracy?
    # Usually single metrics like 'recall' without specifying class imply macro or weighted or positive class.
    # Looking at sample:
    # Class 1 recall is 0.80. Class 0 is 0.40. Output says 0.800. So it's Class 1 (Outcome=1).
    # Class 1 precision is 0.57. Output 0.571.
    # Class 1 f1 is 0.67. Output 0.667.
    # Accuracy is 0.60.
    
    # So we print metrics for pos_label=1.
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)

    print(f"accuracy: {acc:.3f}")
    print(f"recall: {rec:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {prec:.3f}")

if __name__ == "__main__":
    main()
