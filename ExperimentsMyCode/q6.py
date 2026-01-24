import os
import sys
import warnings

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv(os.path.join(sys.path[0], input()))
df["target"] = (df["quality"] >= 7).astype(int)
X = df.drop(columns=["target", "quality"])
y = df["target"]

model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

print("AdaBoost Classifier Results:")
print(f"Accuracy: {accuracy_score(ytest, ypred):.2f}")
print(f"Precision: {precision_score(ytest, ypred, zero_division=0):.2f}")
print(f"Recall: {recall_score(ytest, ypred, zero_division=0):.2f}")
print(f"F1-score: {f1_score(ytest, ypred, zero_division=0):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(ytest, ypred))