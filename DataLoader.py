import os
import pandas as pd

class DataLoader():
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self):
        self.__validate_file_exists()
        
        df = pd.read_csv(self.file_path)
        return df
        
    def __validate_file_exists(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f"El archivo no existe en: {self.file_path}")