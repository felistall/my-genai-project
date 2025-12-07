# src/9_semantic_search.py

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Simple cosine similarity between two vectors.
    Returns a value between -1 and 1 (higher = more similar).
    """
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def main():
    # 1) Embedding model (same as in 8_embeddings_langchain.py)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2) Our "document collection"
    documents = [
        "Retrieval-Augmented Generation (RAG) uses external documents to answer questions.",
        "LoRA and QLoRA are efficient fine-tuning methods for large language models.",
        "I love making pasta with cheese and tomato sauce.",
        "Semantic search finds text with similar meaning using embeddings.",
        "Transformers and attention mechanisms are the core of modern NLP models.",
    ]

    print(f"\nWe have {len(documents)} candidate documents:\n")
    for i, doc in enumerate(documents):
        print(f"[{i}] {doc}")
    print("\n")

    # 3) Get embeddings for all documents
    doc_vectors = np.array(embeddings.embed_documents(documents))

    # 4) Ask user for a query
    query = input("Enter your search query: ")
    query_vector = np.array(embeddings.embed_query(query))

    # 5) Compute cosine similarity between query and each document
    similarities = [cosine_similarity(query_vector, dv) for dv in doc_vectors]

    # 6) Sort documents by similarity (highest first)
    ranked_indices = np.argsort(similarities)[::-1]

    print("\n=== Semantic Search Results ===\n")
    for idx in ranked_indices:
        print(f"Score: {similarities[idx]:.4f}")
        print(f"Text:  {documents[idx]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
