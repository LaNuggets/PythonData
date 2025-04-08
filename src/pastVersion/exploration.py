from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel

import pandas as pd;
import matplotlib.pyplot as plt;


df = pd.read_csv("data/train.csv")

prep = TitanicPreprocessor()
X, y = prep.fit_transform(df)

model = TitanicModel()
model.train(X, y)
  



print("Dimensions du dataset :", df.shape)

print(df.head())


print("Average age is", df["Age"].mean())

df['Age_group'] = df['Age'].apply(lambda x: '-10' if x <= 10 else '+10')
print(df.groupby('Age_group')['Survived'].mean())


df["IsMale"] = (df["Sex"] == "male").astype(int)

df["Sex_binary"] = df["Sex"].map({"male": 0, "female": 1})
print(df[["Sex", "Sex_binary"]].head())

print(df["Sex"].eq("male").mul(100).mean())


from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")

#Séance 2


print(df.isna().sum())