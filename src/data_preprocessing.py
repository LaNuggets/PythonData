import pandas as pd

def preprocess_data(df):
    # Remplacer les NaN par la médiane pour les colonnes numériques
    for column in df.columns:
        if df[column].dtype != "object":
            df[column].fillna(df[column].median(), inplace=True)
    # Encoder les variables catégorielles, etc.
    return df
