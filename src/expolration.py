import pandas as pd;

df = pd.read_csv("train.csv")
print(df.head())

print("Average age is", df["Age"].mean())

#df["IsMale"] = (df["Sex"] == "male").astype(int)

df["Sex_binary"] = df["Sex"].map({"male": 0, "female": 1})
print(df[["Sex", "Sex_binary"]].head())

print(df["Sex"].eq("male").mul(100).mean())

#from ydata_profiling import ProfileReport
 
#profile = ProfileReport(df, title="Profiling Report")
#profile.to_file("your_report.html")