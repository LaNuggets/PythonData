from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel
import pandas as pd


def main():
    
  df = pd.read_csv("data/train.csv")
   
  prep = TitanicPreprocessor()
  X, y = prep.fit_transform(df)
    
  model = TitanicModel()
  X_train, X_test, y_train, y_test = model.train(X, y)
    
  y_pred = model.predict(X_test)
  print("Prédictions :", y_pred[:5])
    
if __name__ == "__main__":
  main()