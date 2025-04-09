import sys
import os
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score

# Ajouter le répertoire 'src' au chemin d'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importer les fonctions nécessaires du fichier modeling.py
from modeling import preprocess_and_split_data, create_pipeline

@pytest.fixture
def data_split():
    """Fixture pour prétraiter et séparer les données en train/test."""
    # Exemple de données
    data = {
        'Age': [22, 38, 26, 35],
        'SibSp': [1, 1, 0, 1],
        'Parch': [0, 0, 0, 0],
        'Fare': [7.25, 71.2833, 7.925, 53.1],
        'Pclass': [3, 1, 3, 1],
        'Sex_male': [1, 0, 0, 1],
        'Embarked_Q': [0, 0, 0, 0],
        'Embarked_S': [1, 1, 1, 1],
        'Survived': [0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    return preprocess_and_split_data(df)

@pytest.fixture
def pipeline():
    """Fixture pour créer un pipeline complet (prétraitement + modèle)."""
    return create_pipeline()

def test_model_accuracy(pipeline, data_split):
    """Test de l'accuracy du modèle avec le pipeline."""
    X_train, X_test, y_train, y_test = data_split
    # Entraîner le pipeline sur les données
    pipeline.fit(X_train, y_train)
    
    # Prédire les résultats
    y_pred = pipeline.predict(X_test)
    
    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Vérifier que l'accuracy est supérieure à un seuil raisonnable
    assert accuracy > 0.7, f"Accuracy trop faible : {accuracy}"

def test_model_coefficients(pipeline, data_split):
    """Test pour vérifier que les coefficients du modèle sont non nuls dans le pipeline."""
    X_train, X_test, y_train, y_test = data_split
    pipeline.fit(X_train, y_train)
    
    # Vérifier que les coefficients du modèle sont non nuls
    assert all(coef != 0 for coef in pipeline.named_steps['classifier'].coef_.flatten()), "Certains coefficients sont nuls."

