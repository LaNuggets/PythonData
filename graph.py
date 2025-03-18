import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
plt.style.use('_mpl-gallery')

x = 0.5 + np.arange(12)
y=[]

resp = df.isna().sum()

for i in resp.iloc:
    y.append(i.tolist())

fig, ax = plt.subplots()

ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 12), xticks=np.arange(1, 12),
       ylim=(0, 12), yticks=np.arange(1, 12))

plt.show()