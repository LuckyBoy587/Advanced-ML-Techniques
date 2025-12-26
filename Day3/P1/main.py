import pandas as pd
import os, sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.simplefilter(action='ignore') 

df = pd.read_csv(os.path.join(sys.path[0], input()))

print("First 5 rows of the dataset:")
print(df.head())

print("Number of samples in the data:")
print(len(df))

print("Data types of each column:")
print(df.dtypes)

feature_columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
print("Feature columns:")
print(feature_columns)

print("Statistical summary of numeric columns:")
print(df.describe())

X = df[feature_columns]
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))