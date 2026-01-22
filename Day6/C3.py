import pandas as pd
import os, sys
import warnings

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, \
    precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(sys.path[0], input()))

print(df.head())

X = df.drop(columns="price_range.enc")
y = df["price_range.enc"]

model = AdaBoostClassifier(
    DecisionTreeClassifier(random_state=42, max_depth=2),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
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
