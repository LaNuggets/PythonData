import sys
import os
import pytest
import pandas as pd

# Ajouter le répertoire 'src' au chemin d'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import preprocess_data  # Assurez-vous que le module est correctement importé depuis src

@pytest.fixture
def sample_data():
    """Fixture pour fournir un jeu de données d'exemple pour le prétraitement."""
    data = {
        'age': [22, 38, 26, 35],
        'sex': ['male', 'female', 'female', 'male'],
        'fare': [7.25, 71.2833, 7.925, 53.1],
        'embarked': ['S', 'C', 'Q', 'S'],
        'survived': [0, 1, 1, 0],
        'cabin': ['C85', 'C123', 'E46', 'B57 B59 B63 B66']
    }
    return pd.DataFrame(data)

def test_preprocess_data_fill_missing_values(sample_data):
    """Test pour vérifier que les valeurs manquantes sont remplies dans les colonnes 'age' et 'embarked'."""
    df_processed = preprocess_data(sample_data)
    
    # Vérifier que la colonne 'age' ne contient plus de valeurs manquantes
    assert df_processed['age'].notna().all(), "Les valeurs manquantes dans 'age' doivent être remplies."
    
    # Vérifier que la colonne 'embarked_Q' ne contient plus de valeurs manquantes
    assert df_processed['embarked_Q'].notna().all(), "Les valeurs manquantes dans 'embarked' doivent être remplies."

def test_preprocess_data_check_data_type(sample_data):
    """Test pour vérifier que les types de données sont corrects après le prétraitement."""
    df_processed = preprocess_data(sample_data)
    
    # Vérifier que la colonne 'sex' est maintenant une variable numérique (binaire)
    assert df_processed['sex'].dtype == 'int64', "'sex' doit être une variable numérique (binaire)."
    
    # Vérifier que la colonne 'embarked_Q' existe après le prétraitement
    assert 'embarked_Q' in df_processed.columns, "La colonne 'embarked_Q' doit exister."

def test_preprocess_data_columns_existence(sample_data):
    """Test pour vérifier l'existence des colonnes après le prétraitement."""
    df_processed = preprocess_data(sample_data)
    
    # Vérifier que la colonne 'survived' existe dans les données traitées
    assert 'survived' in df_processed.columns, "'survived' doit exister après le prétraitement."
    
    # Vérifier que la colonne 'age' existe dans les données traitées
    assert 'age' in df_processed.columns, "'age' doit exister après le prétraitement."
