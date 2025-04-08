import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('train.csv')

# Vérifier les colonnes du dataframe
print("Colonnes du dataframe :")
print(df.columns)

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Traiter les valeurs manquantes :
# - Remplacer les valeurs manquantes des colonnes numériques par la moyenne
# - Remplacer les valeurs manquantes des colonnes catégorielles par le mode (valeur la plus fréquente)

# Remplir les valeurs manquantes pour les colonnes numériques (moyenne)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Remplir les valeurs manquantes pour les colonnes catégorielles (mode)
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Vérification après remplissage des valeurs manquantes
print("\nValeurs manquantes après traitement :")
print(df.isnull().sum())

# Encodage des variables catégorielles (par exemple, 'Sex' et 'Embarked')
df = pd.get_dummies(df, drop_first=True)

# Assurer que la colonne cible existe bien avant de la supprimer
if 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
else:
    print("\nLa colonne 'target' n'existe pas. Utilisation de 'Survived' comme cible.")
    # Utiliser 'Survived' comme cible si 'target' n'existe pas
    X = df.drop('Survived', axis=1)
    y = df['Survived']

# Vérifier que X et y ont bien été créés
print("\nX (features) :")
print(X.head())
print("\ny (target) :")
print(y.head())

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données (moyenne 0 et écart-type 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vérifier les types des données
print("\nTypes des données dans X (features) :")
print(X.dtypes)

# Entraîner un modèle de régression logistique
model = LogisticRegression(max_iter=1000)

# Entraîner le modèle
model.fit(X_train_scaled, y_train)

# Prédire et évaluer
y_pred = model.predict(X_test_scaled)

# Afficher les résultats
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non', 'Oui'], yticklabels=['Non', 'Oui'])
plt.xlabel('Prédictions')
plt.ylabel('Véritables')
plt.title('Matrice de Confusion')
plt.show()

# Afficher le rapport de classification
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optionnel: Afficher la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

# Sauvegarder le modèle entraîné avec joblib
joblib.dump(model, 'model_logistic_regression.pkl')  # Sauvegarde du modèle

# Sauvegarder le scaler utilisé pour la normalisation des données
joblib.dump(scaler, 'scaler.pkl')  # Sauvegarde du scaler

# Charger le modèle et le scaler 
model_loaded = joblib.load('model_logistic_regression.pkl')
scaler_loaded = joblib.load('scaler.pkl')

# *** Prédiction sur de nouvelles données ***

# Exemple de nouvelles données à prédire (vous devez charger ou créer ces données)
# Par exemple, charger un fichier CSV contenant de nouvelles données
# X_new = pd.read_csv('new_data.csv')

# Assurez-vous que les nouvelles données sont prétraitées de la même manière que les données d'entraînement
# Effectuer un encodage similaire pour les variables catégorielles si nécessaire
# X_new = pd.get_dummies(X_new, drop_first=True)

# Normalisez les nouvelles données avec le même scaler que celui utilisé pour entraîner le modèle
# Exemple : si vous avez une nouvelle ligne de données, vous pouvez la transformer ainsi :
X_new = pd.DataFrame({
    'Age': [30],
    'Fare': [7.25],
    'Pclass_2': [1],
    'Pclass_3': [0],
    'Sex_male': [1],
    'Embarked_Q': [0],
    'Embarked_S': [1],
})  # Exemple de nouvelle observation (s'assurer que le DataFrame a les mêmes colonnes que celles d'entraînement)

# Utiliser le modèle pour des prédictions sur de nouvelles données
X_new_scaled = scaler_loaded.transform(X_new)  # X_new est un nouvel ensemble de données à prédire
y_new_pred = model_loaded.predict(X_new_scaled)  

# Afficher les prédictions
print("\nPrédictions pour les nouvelles données :")
print(y_new_pred)




