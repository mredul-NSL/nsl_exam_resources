# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:32:39 2021

@author: Mredul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv(r"C:\Users\DELL\Desktop\Exam_Codes\breast-cancer-wisconsin")
df.head(6).T

df.describe().T
print(df.dtypes)

plt.figure(figsize=(21,21))
plt.title("Pearson Correlation Heatmap")
corr = df.corr(method='pearson')
mask = np.tril(df.corr())
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,
           annot = True, # to show the correlation degree on cell
           vmin=-1,
           vmax=1,
           center= 0,
           fmt='0.2g', #
           cmap= 'coolwarm',
           linewidths=3, # cells partioning line width
           linecolor='white', # for spacing line color between cells
           square=False,#to make cells square
           cbar_kws= {'orientation': 'vertical'}
           )

b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b,t)
plt.show()