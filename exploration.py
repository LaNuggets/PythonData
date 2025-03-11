import pandas as pd

# Chemin absolu vers le fichier CSV
file_path = '/home/christian/PythonData/train.csv'

df = pd.read_csv(file_path)

# Afficher les 5 premières lignes
print(df.head())

# Afficher des informations générales sur les données
print(df.info())