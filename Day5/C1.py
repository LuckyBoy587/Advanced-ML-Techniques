import pandas as pd
import os, sys
import warnings

warnings.simplefilter("ignore")

df = pd.read_csv(os.path.join(sys.path[0], input()))

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

X = df.drop('target', axis=1)
y = df['target']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


accuracies = []
predictions = []
actuals = []

for train_index, val_index in skf.split(X, y):
    X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    # model
    rf = RandomForestClassifier(
        max_depth=5,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_val)
    predictions.extend(y_pred)
    actuals.extend(y_val)

print("=" * 33)
print(f"Accuracy: {accuracy_score(actuals, predictions):.3f}")

final_model = RandomForestClassifier(
    max_depth=5,
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)

final_model.fit(X, y)
oob_score = final_model.oob_score_

print(f"OOB Score: {oob_score:.3f}")
print("=" * 33)