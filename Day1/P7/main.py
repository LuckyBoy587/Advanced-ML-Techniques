import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def main():
    df = pd.read_csv(os.path.join(sys.path[0], input()))
    df = df.select_dtypes(include=['number'])
    X = df.drop(columns=["left"])
    y = df["left"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    print(model)
    
    print(model.predict(X_test))
    
if __name__ == "__main__":
    main()