import pandas as pd
import os, sys
import warnings

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv(os.path.join(sys.path[0], input()))
df["target"] = (df["quality"] >= 7).astype(int)
X = df.drop(columns=["target", "quality"])
y = df["target"]
models = [
    ("Naive Bayes Cross-Validation", GaussianNB()),
    ("Decision Tree Cross-Validation", DecisionTreeClassifier(random_state=42)),
    ("SVM Cross-Validation", SVC(kernel='linear', random_state=42,))
]
kf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
for name, model in models:
    print(f"{name} Results:")
    print(f"Average Accuracy: {cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean():.2f}")
    print(f"Average Precision: {cross_val_score(model, X, y, cv=kf, scoring=make_scorer(precision_score, average='binary', zero_division=0)).mean():.2f}")
    print(f"Average Recall: {cross_val_score(model, X, y, cv=kf, scoring=make_scorer(recall_score, average='binary', zero_division=0)).mean():.2f}")
    print(f"Average F1-score: {cross_val_score(model, X, y, cv=kf, scoring=make_scorer(f1_score, average='binary', zero_division=0)).mean():.2f}")