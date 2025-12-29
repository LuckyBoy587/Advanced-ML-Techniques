import pandas as pd
import numpy as np
import sys
import os
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_scale(X_DT):
    # Select only numeric columns
    numeric_cols = X_DT.select_dtypes(include=['number']).columns
    
    # Initialize Scaler
    scaler = StandardScaler()
    
    # Fit and transform
    X_scaled = X_DT.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_DT[numeric_cols])
    
    return X_scaled
warnings.simplefilter(action='ignore')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ... (Assuming your model 'dt_model' is already trained) ...

def visualize_tree(model, feature_names):
    # Set the size of the plot to be readable
    plt.figure(figsize=(20, 10))
    
    # plot_tree logic
    plot_tree(model, 
              feature_names=feature_names, 
              filled=True, 
              rounded=True, 
              fontsize=10) # Limits depth so the plot doesn't crash your RAM
    
    plt.title("Decision Tree Regressor - HR Analytics (Top 3 Levels)")
    plt.show()

# To call it in your main script:
# visualize_tree(dt_model, X_scaled.columns)
    
def main():
    filename = input().strip()
    filepath = os.path.join(sys.path[0], filename)
    if not os.path.exists(filepath):
        print(f"Error: File '{filename}' not found.")
        return
    df = pd.read_csv(filepath)
    y = df['average_monthly_hours']
    X = df.drop(columns=['average_monthly_hours', 'Department', 'salary'])
    X_scaled = data_scale(X)
    dt_model = DecisionTreeRegressor(random_state=42)
    cv_scores = cross_val_score(dt_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    avg_mse = -np.mean(cv_scores)
    print(f"Cross-validated MSE: {avg_mse}")
    dt_model.fit(X_scaled, y)
    predictions = dt_model.predict(X_scaled)
    print(f"Predictions: {predictions}")
    visualize_tree(dt_model, X_scaled.columns)
    

if __name__ == "__main__":
    main()