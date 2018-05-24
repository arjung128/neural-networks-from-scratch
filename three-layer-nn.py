import pandas as pd
import numpy as np
import sklearn

import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('wine.csv')
df.head()

print(df)
