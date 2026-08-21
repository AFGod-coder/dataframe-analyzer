import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

def newTopic(title, command_action):
  print('\n' + '=' * 50)
  print(title)
  print('=' * 50)
  if callable(command_action):
    result = command_action()
    if result is not None:
      print(result)
  else:
    print(command_action)


newTopic('FIRST ROWS:', lambda: df.head())
newTopic('LAST ROWS:', lambda: df.tail())
newTopic('COLUMN NAMES:', df.columns.tolist())
newTopic('GENERAL INFORMATION (DATA TYPES AND MEMORY USE):', lambda: df.info())
newTopic("NUMBERS ESTADISTIC:", lambda: df.describe())
newTopic('NULLs COUNT:', lambda: df.isnull().sum())

rows, column = df.shape
print(f"TOTAL SIZE: {rows}, ROWS AND {column} COLUMNS")
print(f'TOTAL DUPLICAD ITEMS: {df.duplicated().sum()}')

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df.to_csv('clean_train.csv', index=False, encoding='utf-8')