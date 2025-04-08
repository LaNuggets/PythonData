import sys
import os
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ajouter le répertoire 'src' au chemin d'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modeling import X_train_scaled, X_test_scaled, y_train, y_test

@pytest.fixture
def model():
    """Fixture pour créer et retourner un modèle de régression logistique."""
    return LogisticRegression(max_iter=1000)

def test_model_accuracy(model):
    """Test de l'accuracy du modèle."""
    # Entraîner le modèle
    model.fit(X_train_scaled, y_train)
    
    # Prédire les résultats
    y_pred = model.predict(X_test_scaled)
    
    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Vérifier que l'accuracy est supérieure à un seuil raisonnable
    assert accuracy > 0.7, f"Accuracy trop faible : {accuracy}"

def test_model_coefficients(model):
    """Test pour vérifier que les coefficients du modèle sont non nuls."""
    model.fit(X_train_scaled, y_train)
    # Vérifier que les coefficients du modèle sont non nuls
    assert all(coef != 0 for coef in model.coef_.flatten()), "Certains coefficients sont nuls."

