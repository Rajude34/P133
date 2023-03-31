import pandas as pd
import csv

df = pd.read_csv('total_stars.csv')
df

df.columns

df.drop(['Unnamed: 0','Unnamed: 6', 'Star_name.1', 'Distance.1', 'Mass.1', 'Radius.1','Luminosity'],axis=1,inplace=True)
df

final_data = df.dropna()
final_data

final_data.reset_index(drop=True,inplace = True)
final_data

final_data.to_csv('final_data.csv')
final_data.info()

final_data.describe()

final_data.head(5)

final_data.dtypes