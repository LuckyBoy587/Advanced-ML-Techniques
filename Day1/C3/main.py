import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def main():
    df = pd.read_csv(os.path.join(sys.path[0], input()))
    X = df["HealthText"]
    y = df["Outcome"]
    
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    test_input = "Age group: Senior | BMI status: Overweight | Glucose category: Very High Glucose Level"
    test_vector = vectorizer.transform([test_input])
    print(f"Prediction: {model.predict(test_vector)[0]}")
    
if __name__ == "__main__":
    main()