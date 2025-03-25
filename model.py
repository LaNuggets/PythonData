import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("train.csv") # récupération du csv

df['isMale'] = df["Sex"].apply(lambda x: 1 if x == 'male' else 0) # création d'une nouvelle colonne pour savoir si le passager est un homme ou une femme mais en donné binaire
df['Age'] = df['Age'].fillna(df['Age'].mean()) # remplir les age manquant par l'age moyen

df = df.drop('Name', axis=1) # supprésion de la table "Name"
df = df.drop('Sex', axis=1) # supprésion de la table "Sex"
df = df.drop('Ticket', axis=1) # supprésion de la table "Ticket"
df = df.drop('Cabin', axis=1) # supprésion de la table "Cabin"
df = df.drop('Embarked', axis=1) # supprésion de la table "Embarked"

X = df.drop('Survived', axis=1) # récupération de la table des survivant
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2,
random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) # entrainement du model
y_pred = model.predict(X_test) # résultat des prédiction

acc = accuracy_score(y_test, y_pred) # calcule de l'accuracy
f1 = f1_score(y_test, y_pred, average='binary')

print(acc, f1)
cm = confusion_matrix(y_test, y_pred) # création de matrice de confusion
print(cm)
