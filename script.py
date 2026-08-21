import pandas as pd

df = pd.read_csv('train.csv')

def newTopic(title, command_action):
    print("\n" + "="*50)
    print(title)
    print("="*50)
    
    if callable(command_action):
        result = command_action()
        if result is not None:
            print(result)
    else:
        print(command_action)

newTopic("FIRST ROWs:", lambda: df.head())
newTopic("LAST ROWS:", lambda: df.tail())
newTopic("COLUMN NAMES:", df.columns.tolist())
newTopic("GENERAL INFORMATION (DATA TYPES AND MEMORY USE ):", lambda: df.info())
newTopic("NULLs COUNT:", lambda: df.isnull().sum())