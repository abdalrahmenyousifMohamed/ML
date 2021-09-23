#effects-of-covid-19-on-trade-at-21-July-2021-provisional.csv
import random
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
#import seaborn as sns
data=pd.read_csv("effects-of-covid-19-on-trade-at-21-July-2021-provisional.csv")
print(data.info())
print(data)
print(data.isnull().sum())#;print(data.notnull().sum())
#print(data.columns[data.isna().any()])
#print(data.isna())
#data.drop(['Revenue (Millions)', 'Metascore'],axis=1,inplace=True)
#data.dropna() # remove rows
#data.dropna(axis=1) # remove columns
#data['Commodity'].fillna("foulll",inplace=True)
#print(data.columns[data.isna().any()])
cols=data.loc[:,["Cumlative","Value"]]
print(cols)
my_imputer=SimpleImputer()
imputed_values=my_imputer.fit_transform(cols)
print(imputed_values)
new_values=pd.DataFrame(imputed_values)# array to dataframe
new_values.columns=cols.columns # name of columns to new columns values
print(new_values.isnull().sum())
data['Value']=new_values['Value']
data['Cumlative']=new_values['Cumlative']
print(data)