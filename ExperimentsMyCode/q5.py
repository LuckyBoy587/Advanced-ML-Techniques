import os, sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(os.path.join(sys.path[0], input()))
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df["Embarked"].fillna("S", inplace=True)
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
df["Embarked"] = le.fit_transform(df["Embarked"])
df["Age"].fillna(df["Age"].median(), inplace=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

print("Random Forest Classifier Results:")
print(f"Accuracy: {accuracy_score(ytest, ypred):.2f}")
print(f"Precision: {precision_score(ytest, ypred, zero_division=0):.2f}")
print(f"Recall: {recall_score(ytest, ypred, zero_division=0):.2f}")
print(f"F1-score: {f1_score(ytest, ypred, zero_division=0):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(ytest, ypred))