#-*- coding:utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
sns.set(color_codes=True)
pm2_5 = pd.read_csv('C:\\Users\\Bai\\Desktop\\data\\pm25.csv')
data = pm2_5.iloc[:,1].dropna()
sns.distplot(data)
plt.show()