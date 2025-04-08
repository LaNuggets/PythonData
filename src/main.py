from preprocessor import TitanicPreprocessor
from TitanicModel import TitanicModel
import pandas as pd

def main():
    df = pd.read_csv("data/train.csv")

    prep = TitanicPreprocessor()
    X, y = prep.fit_transform(df)

    model = TitanicModel()
    model.train(X, y)

    y_pred = model.predict(X)
    print("Prédictions :", y_pred[:5])
    
if __name__ == "__main__":
    main()
