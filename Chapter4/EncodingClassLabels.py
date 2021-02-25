import pandas as pd
import numpy as np

df = pd.DataFrame([
    ['green','M',10.1,'class2'],
    ['red','L',13.5,'class1'],
    ['blue','XL','15.3','class2']
])

df.columns=['color','size','price','classLabel']
class_mapping =  {label:idx for idx, label in enumerate(np.unique(df['classLabel']))}

print(df)
print(class_mapping)

df['classLabel'] = df['classLabel'].map(class_mapping)
print(df)