from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

def custom_scaling(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def custom_train_test_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

def evaluate_classification(y_test, y_pred):
    # Confusion Matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================\n")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("===================\n")

    # Individual Metrics
    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"precision: {precision_score(y_test, y_pred, pos_label=1):.3f}")
