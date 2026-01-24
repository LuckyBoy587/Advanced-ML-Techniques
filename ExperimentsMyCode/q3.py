import os, sys
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
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

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
ypred = model.fit(xtrain, ytrain).predict(xtest)
cm = confusion_matrix(ytest, ypred)
print("Decision Tree Results:")
print(f"Precision: {precision_score(ytest, ypred):.2f}")
print(f"Recall (Sensitivity): {recall_score(ytest, ypred):.2f}")
print(f"Specificity: {(cm[0, 0] / cm[0].sum()):.2f}")
print(f"AUC-ROC: {roc_auc_score(ytest, model.predict_proba(xtest)[:, 1]):.2f}")