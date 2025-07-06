#!/usr/bin/env python3
from typing import List
import numpy as np
import os
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer
import umap.umap_ as umap
import matplotlib.pyplot as plt

from utils import load_dataset  # ajuste o import se seu utilitário tiver outro caminho
import csv

def save_metrics(title, n_clusters, n_labels, ari, nmi, hom):
    os.makedirs("results", exist_ok=True)
    file_path = "results/clustering_metrics.csv"
    header = ["filename", "k_found", "k_true", "ARI", "NMI", "Homogeneity"]

    # Se o arquivo ainda não existe, crie com cabeçalho
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([title, n_clusters, n_labels, ari, nmi, hom])


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def embed_sentences(sentences: List[str], tokenizer: BertTokenizer, model: BertModel, device: str) -> np.ndarray:
    """Retorna um np.ndarray (N, hidden_size) com o embedding [CLS] de cada sentença."""
    embeddings = []

    for txt in sentences:
        inputs = tokenizer(
            txt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_emb)

    return np.vstack(embeddings)


def find_optimal_k(embeddings: np.ndarray, max_k: int = 15) -> int:
    """
    Encontra o número ótimo de clusters (k) usando o método da Silhueta.
    """
    print(f"\nBuscando k ótimo entre 2 e {max_k} usando Análise de Silhueta...")
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
        print(f"  k={k}, Silhouette Score: {score:.4f}")

    # Plota os scores
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, silhouette_scores, "bx-")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Análise de Silhueta para Encontrar k Ótimo")
    plt.show()

    # Encontra o k com o maior score de silhueta
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"-> k ótimo encontrado: {optimal_k}\n")
    
    return optimal_k

def evaluate_clustering(embeddings: np.ndarray, labels: List[str], title: str, optimal_k: int) -> None:
    """Clusteriza com k ótimo, imprime métricas e compara com labels reais."""
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    num_real_labels = len(le.classes_)
    
    # Usa o k ótimo encontrado
    n_clusters = optimal_k

    print(f"=== {title} ===")
    print(f"Número de Labels (Diagnósticos Reais): {num_real_labels}")
    print(f"Número de Clusters (k Ótimo Encontrado): {n_clusters}")
    print("-" * 40)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    y_pred = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    hom = homogeneity_score(y_true, y_pred)

    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info  (NMI): {nmi:.4f}")
    print(f"Homogeneity            : {hom:.4f}\n")

    # Visualização
    reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{title}\n(k_encontrado={n_clusters} vs k_real={num_real_labels})", fontsize=14)

    axes[0].scatter(reduced[:, 0], reduced[:, 1], c=y_true, cmap="tab10", s=10)
    axes[0].set_title("True Labels")

    axes[1].scatter(reduced[:, 0], reduced[:, 1], c=y_pred, cmap="tab10", s=10)
    axes[1].set_title("Cluster Assignments")

    plt.tight_layout()
    plt.savefig(f"results/umap_{title}.png", dpi=400)
    plt.show()
    plt.close()

    save_metrics(title, n_clusters, num_real_labels, ari, nmi, hom)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    data_dir = "/home/wytcor/PROJECTs/EDA-sentences/data/vlms-and-llms-sentences"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"Nenhum arquivo CSV encontrado em {data_dir}")
        return

    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device).eval()

    for csv_name in csv_files:
        csv_path = os.path.join(data_dir, csv_name)
        df = load_dataset.load_data(file_folder_path=csv_path)

        sentences = df["sentence"].tolist()
        diagnostics = df["diagnostic"].tolist()

        print(f"\nProcessando {csv_name} ({len(sentences)} frases)...")
        embeddings = embed_sentences(sentences, tokenizer, model, device)
        print(f"Embeddings prontos: {embeddings.shape}")

        # 1. Encontra o k ótimo a partir dos dados
        optimal_k = find_optimal_k(embeddings, max_k=15)
        
        # 2. Usa o k ótimo para avaliar e comparar
        title = os.path.splitext(csv_name)[0]
        evaluate_clustering(embeddings, diagnostics, title, optimal_k)

if __name__ == "__main__":
    main()
