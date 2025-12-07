# src/8_embeddings_langchain.py

from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    # 1) Choose an embedding model
    # This one is small, fast, and good for semantic search
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2) Some example texts
    texts = [
        "Retrieval-Augmented Generation helps language models use external knowledge.",
        "I am building a GenAI project with RAG, LoRA, and QLoRA.",
        "This is a random sentence about cooking pasta.",
        "RAG combines a retriever and a generator to answer questions from documents.",
    ]

    print(f"\nEncoding {len(texts)} texts into embeddings...\n")
    vectors = embeddings.embed_documents(texts)

    # 3) Show basic info
    print("Number of embeddings:", len(vectors))
    print("Embedding dimension:", len(vectors[0]))

    # 4) Preview a few numbers of the first vector
    print("\nFirst embedding vector (first 10 values):")
    print(vectors[0][:10])

    # 5) Link text to vector visually
    print("\nText ↔ Embedding preview:")
    for t, v in zip(texts, vectors):
        print("-" * 60)
        print("Text:", t)
        print("Vector snippet:", v[:5])  # show only first 5 numbers


if __name__ == "__main__":
    main()
