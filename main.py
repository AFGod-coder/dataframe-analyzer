from DataLoader import DataLoader
from utils import clean_dataframe


if __name__ == "__main__":
    try:
        
        loader = DataLoader(file_path="train.csv") 
        clean_df = clean_dataframe(loader.load())
        
    except FileNotFoundError as e:
        print(f"File error: {e}")   
    except Exception as e:
        print(f"Fatal error ocurred: {e}")   
        