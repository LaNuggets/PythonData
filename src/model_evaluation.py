from sklearn.metrics import confusion_matrix, classification_report

def evaluate_classification(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
