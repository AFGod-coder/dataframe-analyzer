import pandas as pd

df = pd.read_csv('train.csv')

print("="*50)
print("FIRST ROWs:")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("LAST ROWS:")
print("="*50)
print(df.tail())

print("\n" + "="*50)
print("COLUMN NAMES:")
print("="*50)
print(df.columns.tolist())

