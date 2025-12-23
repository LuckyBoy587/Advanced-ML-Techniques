import pandas as pd
import sys
import os
import warnings
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.simplefilter(action='ignore')

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
        # Although not explicitly asked for "empty or invalid" text in this prompt's constraints, 
        # previous problem had it. The current constraints say:
        # "If the CSV file is not found, the program prints: 'Error: File '{filename}' not found.' and exits."
        # It implies standard read. 
        # I will stick to the specific constraints of P6 which focus on file not found.
        # But if it crashes on read, it's good to handle.
        sys.exit()

    # 1. Creating dummy variables for salary
    print("Creating dummy variables for salary:")
    dummies_salary = pd.get_dummies(df.salary, prefix='salary')
    # Concatenate to the right
    df_with_salary = pd.concat([df, dummies_salary], axis=1)
    
    print(df_with_salary.head())

    # 2. Creating dummy variables for department
    print("Creating dummy variables for department:")
    dummies_dept = pd.get_dummies(df.Department, prefix='dept')
    # Concatenate to the right of the PREVIOUS dataframe (df_with_salary)
    df_final = pd.concat([df_with_salary, dummies_dept], axis=1)
    
    print(df_final.head())

    # 3. Final dataframe with dummy variables
    print("Final dataframe with dummy variables:")
    print(df_final.head())

    # 4. Training and test dataset sizes
    # Split data into training and testing sets
    # "Uses train_test_split() from scikit-learn with train_size=0.7"
    # "The split is performed on the entire dataframe (X) including all dummy variables."
    # Wait, usually X is features and y is target. The prompt says "Divide the complete dataset... into Training set... Test set"
    # And then "Further separate the features (X) from the target variable (left)"
    
    # So first split the whole dataframe
    train_set, test_set = train_test_split(df_final, train_size=0.7, shuffle=False) 
    # Note: "Random state is not specified". 
    # However, sample output shows consistent sizes. 
    # Sample output sizes: Train (36, 17), Test (16, 17) for a 52 row file (Sample.csv has 52 rows).
    # 52 * 0.7 = 36.4 -> 36. 
    # train_test_split default shuffle is True. 
    # If I don't set random_state, the rows selected will change, but sizes will be constant.
    
    print(f"Size of training dataset: {train_set.shape}")
    print(f"Size of test dataset: {test_set.shape}\n")

    # 5. Shapes of input/output features after train-test split
    # Separate features (X) from the target variable (left)
    
    X_train = train_set.drop(columns='left')
    y_train = train_set['left']
    
    X_test = test_set.drop(columns='left')
    y_test = test_set['left']
    
    print("Shapes of input/output features after train-test split:")
    print(f"{X_train.shape} {y_train.shape} {X_test.shape} {y_test.shape}")

if __name__ == "__main__":
    main()
