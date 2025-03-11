import pandas as pd

df = pd.read_csv("train.csv")
# df['isMale'] = df["Sex"].apply(lambda x: 1 if x == 'male' else 0) # add column isMale 1 or 0
# print(df.groupby('Sex')["Survived"].mean())

# df['Age_group'] = df['Age'].apply(lambda x: '-10' if x <= 10 else '+10') # add columns and get the mean of people who survived under 10 year old
# print(df.groupby('Age_group')['Survived'].mean())


# print(df["Sex"].eq("male").mean()) # get male percentage
