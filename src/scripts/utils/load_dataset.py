import numpy as np
import pandas as pd

def load_data(file_folder_path:str):
    try:
        data = pd.read_csv(filepath_or_buffer=file_folder_path, sep=",")
        return data
    except FileNotFoundError:
        print("Arquivo não encontrado!")