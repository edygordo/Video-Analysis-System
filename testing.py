import pandas as pd

my_data = pd.read_csv('spatial.csv',sep=',')
print(my_data.index.name)