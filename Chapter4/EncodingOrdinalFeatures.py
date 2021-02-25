import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.DataFrame([['green', 'M', 10.1,'class2'],['red', 'L', 13.5,'class1'],['blue', 'XL', 15.3,'class2']])
df.columns = ['color', 'size', 'price','classlabel']

df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L','XL'} else 0)
df['x > L'] = df['size'].apply(lambda x:1 if x == 'XL' else 0)
del df['size']
print(df)