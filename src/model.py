from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import pandas as pd

class TitanicModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)
        self.model.fit(X_train, y_train)
        return X_train, X_test, y_train, y_test
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def evaluate(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred) # calcule de l'accuracy
        f1 = f1_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred) # création de matrice de confusion
        print(cm)
        print("F1-score {f1}")
        print("The accuracy of the model is {acc}")

        return acc