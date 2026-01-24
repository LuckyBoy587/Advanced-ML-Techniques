import os, sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv(os.path.join(sys.path[0], input()))
labels = df["species"].unique()
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])
X = df.drop(columns="species")
y = df["species"]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for name, model in [("SVM (Linear Kernel)", SVC(kernel="linear")), ("SVM (RBF Kernel)", SVC(kernel="rbf"))]:
    print(name)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    print(f"Accuracy: {accuracy_score(ytest, ypred):.2f}")
    print(f"Precision (macro): {precision_score(ytest, ypred, average='macro'):.2f}")
    print(f"Recall (macro): {recall_score(ytest, ypred, average='macro'):.2f}")
    print(f"F1-Score (macro): {f1_score(ytest, ypred, average='macro'):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(ytest, ypred))
    print("Classification Report:")
    print(classification_report(ytest, ypred, digits=3, target_names=labels))