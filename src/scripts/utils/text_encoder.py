from transformers import BertTokenizer, BertModel
import torch

def text_encoder_with_bert(text_to_be_used: str):
    try:
        # Load pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        # Define the text
        text = "The quick brown fox jumps over the lazy dog."
        # Tokenize the text
        inputs = tokenizer(text, return_tensors='pt')
        # Obtain the embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the last hidden state (embeddings)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
    except Exception as e:
        print(f"Erro ao encodar o texto! Erro:{e}\n")
        return None