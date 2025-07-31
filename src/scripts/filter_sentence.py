import os
from utils import load_dataset, filter_diag

if __name__=="__main__":
    data_dir = "/home/wytcor/PROJECTs/EDA-sentences/data/vlms-and-llms-sentences/"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"Nenhum arquivo CSV encontrado em {data_dir}")

    for csv_name in csv_files:
        csv_path = os.path.join(data_dir, csv_name)
        df = load_dataset.load_data(file_folder_path=csv_path)

        sentences = df["sentence"].tolist()
        diagnostics = df["diagnostic"].tolist()

        print(f"Senteça:{filter_diag(sentences)}\n")