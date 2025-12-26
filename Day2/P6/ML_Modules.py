from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

def data_scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def evaluate_classifier(y_test, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================")
    
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("===================")
    
    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"precision: {precision_score(y_test, y_pred, pos_label=1):.3f}")
