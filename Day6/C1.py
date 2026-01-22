import pandas as pd
import os, sys
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(sys.path[0], input()))
print(df.head())
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model = AdaBoostClassifier(random_state=42)
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
}
X = df.drop(columns="Diabetic")
y = df["Diabetic"]
cv_results = cross_validate(
    model, X, y, cv=rkf, scoring=scoring, return_train_score=False
)
print("Confusion Matrix")
from sklearn.metrics import confusion_matrix
import numpy as np
model.fit(X, y)
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
print(cm)
print("===================")
print("Classification Report:")
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
print("===================")
print(f"accuracy:  {cv_results['test_accuracy'].mean():.3f}")
print(f"recall: {cv_results['test_recall'].mean():.3f}")
print(f"f1-score: {cv_results['test_f1'].mean():.3f}")
print(f"precision: {cv_results['test_precision'].mean():.3f}")