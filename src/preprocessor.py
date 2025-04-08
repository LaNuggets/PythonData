import pandas as pd

class TitanicPreprocessor:
    def __init__(self):
       self.df = pd.read_csv("data/train.csv")

    def fit(self, df):
      self.age_moyen= df['Age'].mean()

    def transform(self, df):
      df['isMale'] = df["Sex"].apply(lambda x: 1 if x == 'male' else 0) # création d'une nouvelle colonne pour savoir si le passager est un homme ou une femme mais en donné binaire
      df['Age'] = df['Age'].fillna(self.age_moyen) # remplir les ages manquant par l'age moyen
      df = df.drop('Name', axis=1) # suppression de la table "Name"
      df = df.drop('Sex', axis=1) # suppression de la table "Sex"
      df = df.drop('Ticket', axis=1) # suppression de la table "Ticket"
      df = df.drop('Cabin', axis=1) # suppression de la table "Cabin"
      df = df.drop('Embarked', axis=1) # suppression de la table "Embarked"
      
      X = df.drop('Survived', axis=1) # récupération de la table des survivants
      y = df['Survived']
      
      return X, y

    def fit_transform(self, df):
      self.fit(df)
      X, y = self.transform(df)
      return self.transform(df)
