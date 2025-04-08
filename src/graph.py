import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
plt.style.use('_mpl-gallery')

# make data:
x = 0.5 + np.arange(12)
y=[]

resp = df.isna().sum()

# for i in resp.iloc:
#     y.append(i)
# print(y)

y = [resp.iloc[0], resp.iloc[1], resp.iloc[2], resp.iloc[3], resp.iloc[4], resp.iloc[5], resp.iloc[6], resp.iloc[7], resp.iloc[8], resp.iloc[9], resp.iloc[10], resp.iloc[11]]

# plot
fig, ax = plt.subplots()

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 12), xticks=np.arange(1, 12),
       ylim=(0, 12), yticks=np.arange(1, 12))

plt.show()