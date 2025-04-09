import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame):
    """
    Prétraiter les données : remplir les valeurs manquantes, effectuer les transformations nécessaires,
    et préparer les caractéristiques pour l'entraînement du modèle.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        Les données d'entrée (par exemple, le DataFrame des passagers du Titanic).
    
    Retourne :
    ---------
    X : pd.DataFrame
        Les caractéristiques prêtes pour l'entraînement.
    
    y : pd.Series
        La cible (survived).
    """
    
    # Convertir toutes les colonnes en minuscules pour éviter les erreurs liées à la casse
    df.columns = df.columns.str.lower()

    # Convertir les colonnes booléennes en entiers (0 ou 1)
    bool_columns = df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    
    # Gérer les valeurs manquantes
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(np.median(df['age']))
    
    if 'embarked' in df.columns:
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # Suppression de la colonne 'Cabin' (trop de valeurs manquantes)
    if 'cabin' in df.columns:
        df = df.drop('cabin', axis=1)
    
    # Suppression de colonnes inutiles, comme 'name', 'ticket', etc.
    unnecessary_columns = ['name', 'ticket']
    df = df.drop(columns=[col for col in unnecessary_columns if col in df.columns])

    # Suppression de la colonne 'passengerid' si elle existe
    if 'passengerid' in df.columns:
        df = df.drop(columns=['passengerid'])

    # Conversion des variables catégorielles en variables numériques (One-Hot Encoding)
    if 'embarked' in df.columns:
        df = pd.get_dummies(df, columns=['embarked'], prefix='embarked', drop_first=True)
    
    if 'sex' in df.columns:
        # Création de la colonne 'sex_male' (1 pour 'male', 0 pour 'female')
        df['sex_male'] = df['sex'].map({'male': 1, 'female': 0})
        df = df.drop(columns=['sex'])

    # Séparer les caractéristiques (X) et la cible (y)
    X = df.drop('survived', axis=1) 
    y = df['survived']

    return X, y
