import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame([
    ['green','M',10.1,'class2'],
    ['red','L',13.5,'class1'],
    ['blue','XL','15.3','class2']
])

df.columns=['color','size','price','classLabel']

class_le = LabelEncoder()
y = class_le.fit_transform(df['classLabel'].values)
print(y)

y = class_le.inverse_transform(y)
print(y)

