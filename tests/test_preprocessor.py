import sys
import os
import pytest
import pandas as pd

# Ajouter le répertoire 'src' au chemin d'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import preprocess_data

@pytest.fixture
def sample_data():
    """Fixture pour créer un DataFrame d'exemple."""
    data = {
        'age': [25, 30, None, 40],
        'sex': ['male', 'female', 'female', 'male'],
        'fare': [7.25, 71.2833, None, 8.05]
    }
    return pd.DataFrame(data)

def test_preprocess_data_fill_missing_values(sample_data):
    """Test pour vérifier le remplissage des valeurs manquantes."""
    # Appliquer le prétraitement
    df_processed = preprocess_data(sample_data)
    
    # Vérifier que les NaN dans les colonnes numériques sont remplis par la médiane
    assert df_processed['age'].isnull().sum() == 0, "Il reste des NaN dans la colonne 'age'"
    assert df_processed['fare'].isnull().sum() == 0, "Il reste des NaN dans la colonne 'fare'"
    
    # Vérifier que les NaN dans les colonnes catégorielles sont remplis par la valeur la plus fréquente
    assert df_processed['sex'].isnull().sum() == 0, "Il reste des NaN dans la colonne 'sex'"

def test_preprocess_data_check_data_type(sample_data):
    """Test pour vérifier les types de données après prétraitement."""
    df_processed = preprocess_data(sample_data)
    
    # Vérifier que les colonnes sont des types attendus
    assert df_processed['age'].dtype == 'float64', "La colonne 'age' n'est pas de type 'float64'"
    assert df_processed['fare'].dtype == 'float64', "La colonne 'fare' n'est pas de type 'float64'"
    assert df_processed['sex'].dtype == 'object', "La colonne 'sex' n'est pas de type 'object'"
