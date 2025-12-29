import pandas as pd
import numpy as np
import os, sys

df = pd.read_csv(os.path.join(sys.path[0], input()))
print(df.head())
X = df.drop(columns=["target"])
y = df["target"]


from sklearn.model_selection import KFold, cross_val_score, train_test_split, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = cross_val_score(
    DecisionTreeClassifier(random_state=42, max_depth=4),
    X,
    y,
    cv=kf,
)
    
print(f"K-Fold Accuracy Scores: {accuracies}")
print(f"Mean CV Accuracy: {np.mean(accuracies):.3f}")
model = DecisionTreeClassifier(random_state=42, max_depth=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Hold-Out Method Accuracy: {accuracy_score(y_test, y_pred):.3f}")

loo = LeaveOneOut()
accuracies_loo = cross_val_score(
    DecisionTreeClassifier(random_state=42, max_depth=4),
    X,
    y,
    cv=loo,
)

print(f"LOOCV Accuracy: {np.mean(accuracies_loo):.3f}")

accuracies = cross_val_score(
    DecisionTreeClassifier(random_state=42, max_depth=4),
    X,
    y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
)
print(f"Accuracy: {np.mean(accuracies):.3f}")