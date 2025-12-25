import pandas as pd

df = pd.read_csv('U3-Projects/lawdata.csv')

df.drop_duplicates(inplace=True, keep='first')
df.drop_duplicates(subset='sentence', keep=False, inplace=True)

df.to_csv('U3-Projects/lawdata.csv', index=False)