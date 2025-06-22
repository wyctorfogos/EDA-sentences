import os
import matplotlib.pyplot as plt

def join_images(file_folder_path:str, list_images:dict, result_image_name:str):
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6)) # Aumentei o figsize para melhor visualização
        
        # Carrega e exibe a primeira imagem (LLM)
        llm_image_path = os.path.join(file_folder_path, list_images.get("llm"))
        if os.path.exists(llm_image_path):
            img_llm = plt.imread(llm_image_path)
            ax1.imshow(img_llm)
            ax1.set_title("LLM")
            ax1.axis('off') # Remove os eixos para uma visualização mais limpa
        else:
            print(f"Aviso: Imagem LLM não encontrada em {llm_image_path}")

        # Carrega e exibe a segunda imagem (VLM)
        vlm_image_path = os.path.join(file_folder_path, list_images.get("vlm"))
        if os.path.exists(vlm_image_path):
            img_vlm = plt.imread(vlm_image_path)
            ax2.imshow(img_vlm)
            ax2.set_title("VLM")
            ax2.axis('off') # Remove os eixos para uma visualização mais limpa
        else:
            print(f"Aviso: Imagem VLM não encontrada em {vlm_image_path}")

        # Carrega e exibe a terceira imagem (LLM e VLM)
        llm_and_vlm_image_path = os.path.join(file_folder_path, list_images.get("llm-and-vlm"))
        if os.path.exists(llm_and_vlm_image_path):
            img_llm_and_vlm = plt.imread(llm_and_vlm_image_path)
            ax3.imshow(img_llm_and_vlm)
            ax3.set_title("LLM e VLM")
            ax3.axis('off') # Remove os eixos para uma visualização mais limpa
        else:
            print(f"Aviso: Imagem LLM e VLM não encontrada em {llm_and_vlm_image_path}")

        plt.tight_layout() # Ajusta o layout para evitar sobreposição de títulos e eixos
        plt.savefig(result_image_name) # Salva a imagem combinada
        plt.show()         
    except Exception as e:
        raise ValueError(f"Erro ao juntar as imagens! Erro: {e}\n")
    
if __name__=="__main__":
    file_folder_path="./results"
    list_images={"llm":"tsne_sentence_embeddings_metadata_with_sentences_new-prompt-deepseek-r1:70b.png", 
                 "vlm": "tsne_sentence_embeddings_metadata_with_sentences_of_image-description_qwen2.5:72b.png", 
                 "llm-and-vlm":"tsne_sentence_embeddings_metadata_with_sentences_of_patient_description_and_image-description_llm-deepseek-r1:70b_vllm-qwen2.5:72b.png"}
    
    # Certifique-se de que a pasta 'results' existe
    if not os.path.exists(file_folder_path):
        os.makedirs(file_folder_path)
        print(f"Pasta '{file_folder_path}' criada.")

    result_image_name="./results/jointed_tsne_results/jointed_tsne_results.png"
    join_images(file_folder_path=file_folder_path, list_images=list_images, result_image_name=result_image_name)