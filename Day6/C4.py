import os
import sys
import warnings

import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(sys.path[0], input()))

X = df.drop(columns="price_range.enc").values
y = df["price_range.enc"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

estimators = [
    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ("knn", KNeighborsClassifier()),
    ("dt", DecisionTreeClassifier(random_state=42))
]

final_estimator = LogisticRegression(max_iter=1000, random_state=42)
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("===================")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("===================")




print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"f1-score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(f"precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")