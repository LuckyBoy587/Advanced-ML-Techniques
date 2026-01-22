import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(sys.path[0], input()))
X = df.drop(columns="Diabetic").values
y = df["Diabetic"].values
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)


# --------------------------------------------------
# Stacking Classifier Builder
# --------------------------------------------------
def get_stacking_cls():
    base_estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]

    meta_model = LogisticRegression(max_iter=1000)

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5
    )

    return model


# --------------------------------------------------
# Model Training with Repeated K-Fold
# --------------------------------------------------
def model_stack_classifier(X_cls, y_cls):
    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in rkf.split(X_cls):
        X_train, X_test = X_cls[train_idx], X_cls[test_idx]
        y_train, y_test = y_cls[train_idx], y_cls[test_idx]

        model = get_stacking_cls()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_st = np.concatenate(y_true_all)
    y_pred_st = np.concatenate(y_pred_all)

    acc = accuracy_score(y_true_st, y_pred_st)
    print(f"Accuracy: {acc:.3f}\n")

    return y_true_st, y_pred_st


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------
def evaluate_multilabel_classifier(y_true, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("===================\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=2))
    print("===================")

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')

    print(f"accuracy: {acc:.3f}")
    print(f"recall: {rec:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {prec:.3f}")

evaluate_multilabel_classifier(*model_stack_classifier(X, y))