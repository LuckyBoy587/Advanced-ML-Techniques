import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

def main():
    df = pd.read_csv(os.path.join(sys.path[0], input()))
    df = df.select_dtypes(include=['number'])
    X = df.drop(columns=["left"])
    y = df["left"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, ypred))
    print("===================")
    print("Classification Report:")
    print(classification_report(y_test, ypred))
    print("===================")
    
    print(f"accuracy: {accuracy_score(y_test, ypred):.3f}")
    print(f"recall: {recall_score(y_test, ypred):.3f}")
    print(f"f1-score: {f1_score(y_test, ypred):.3f}")
    print(f"precision: {precision_score(y_test, ypred):.3f}")
    print("code is available inside the 'ML_Modules.py' file")
    
if __name__ == "__main__":
    main()