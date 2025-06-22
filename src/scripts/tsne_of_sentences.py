from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import utils.load_dataset as load_dataset_script
# import utils.text_encoder as text_encoder
import itertools 

def main(csv_file_folder_path:str, device:str="cuda"):
    try:
        dataset = load_dataset_script.load_data(file_folder_path=csv_file_folder_path)
        # Load pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        # Define the text
        for _, sentence_sample in dataset.iterrows():
            # Tokenize the text
            sentence_sample_text=sentence_sample["sentence"].to(device)
            inputs = tokenizer(sentence_sample_text, return_tensors='pt').to(device)
            # Obtain the embeddings
            with torch.no_grad():
                outputs = model(**inputs).to(device)

            # Extract the last hidden state (embeddings)
            last_hidden_states = outputs.last_hidden_state
            # Sentence details
            print(f"Sentence diagnostic:{sentence_sample['diagnostic']}\n")
            print(f"Sentence_text:{sentence_sample['sentence']}\n")
            print(f"Sentence_text_embbedings: {last_hidden_states}\n")
    except Exception as e:
        print(f"Erro ao encodar o texto! Erro:{e}\n")
        return None
    
if __name__=="__main__":
    file_path="/home/wytcor/PROJECTs/EDA-sentences/data/vllms-and-llms-sentences/metadata_with_sentences_of_patient_description_and_image-description_llm-deepseek-r1:70b_vllm-gemma3:27b.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    main(csv_file_folder_path=file_path, device=device)