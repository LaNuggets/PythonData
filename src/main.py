from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel
import pandas as pd

def main():
    # Chargement des données
    df = pd.read_csv("data/train.csv")
    # Prétraitement
    prep = TitanicPreprocessor()
    X, y = prep.fit_transform(df)
    # Entraînement du modèle
    model = TitanicModel()
    model.train(X, y)
    # Prédiction sur les mêmes données pour l'exemple
    y_pred = model.predict(X)
    print("Prédictions :", y_pred[:5])
    
if __name__ == "__main__":
    main()
