import pandas as pd
from ydata_profiling import ProfileReport

# Chemin absolu vers le fichier CSV
file_path = '/home/christian/PythonData/train.csv'

# Charger les données
df = pd.read_csv(file_path)

# Afficher des premières lignes et des informations générales
print("\nAperçu des premières lignes:")
print(df.head())
print("\nInformations générales:")
print(df.info())

# Analyser les valeurs manquantes
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print("\nValeurs manquantes par colonne:")
print(missing_values)
print(f"\nNombre total de valeurs manquantes : {total_missing}")

# Supprimer les colonnes avec trop de valeurs manquantes avec un seuil de 40%
threshold = 0.4
drop_cols = missing_values[missing_values > threshold * len(df)].index
df.drop(columns=drop_cols, inplace=True)

# Supprimer les lignes contenant des valeurs manquantes
df.dropna(inplace=True)

# Imputer des valeurs manquantes restantes par médiane pour les numériques et mode pour les catégoriques
for column in df.columns:
    if df[column].dtype == "object":  # Colonne catégorielle
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Colonne numérique
        df[column].fillna(df[column].median(), inplace=True)

print("\nDataframe après traitement des valeurs manquantes:")
print(df.info())

# Générer le rapport HTML
profile = ProfileReport(df, title="Rapport d'Exploration des Données", explorative=True)
report_path = "/home/christian/PythonData/report.html"
profile.to_file(report_path)

print(f"\nRapport HTML généré: {report_path}")