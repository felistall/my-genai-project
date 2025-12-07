# src/10_rag_pipeline.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PDF_PATH = "data/rag_explained.pdf"          # make sure this file exists
MODEL_PATH = "./outputs/distilgpt2_finetuned"  # your fine-tuned local model


def build_vectorstore(pdf_path: str):
    """Load the PDF, split into chunks, and build FAISS vector store."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {os.path.abspath(pdf_path)}")

    print(f"📄 Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Created {len(chunks)} text chunks")

    # Embeddings model
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"🧠 Loading embeddings model: {embed_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

    # Build FAISS index
    print("📦 Building FAISS vector store...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    print("✅ FAISS index ready.")

    return vectordb


def load_llm(model_path: str):
    """Load your fine-tuned local LLM (distilgpt2_finetuned)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model folder not found at: {os.path.abspath(model_path)}"
        )

    print(f"🧩 Loading LLM from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_answer(question: str, context: str, tokenizer, model, max_new_tokens: int = 200) -> str:
    """
    Generate an answer using the LLM, grounded in the given context.
    """
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", 1)[1].strip()
    else:
        answer = full_text.strip()

    return answer


def main():
    # 1) Build vector store from PDF
    vectordb = build_vectorstore(PDF_PATH)

    # 2) Load fine-tuned LLM
    tokenizer, model = load_llm(MODEL_PATH)

    print("\n🤖 Mini RAG assistant ready!")
    print("Ask questions about the PDF. Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ")
        if question.lower().strip() in ["exit", "quit"]:
            break

        # 3) Retrieve relevant chunks
        docs = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # 4) Generate answer
        answer = generate_answer(question, context, tokenizer, model)

        print("\n--- Retrieved Context (top 3 chunks) ---")
        for i, d in enumerate(docs):
            print(f"[Chunk {i+1}]")
            print(d.page_content[:300], "...\n")

        print("💬 Answer:")
        print(answer)
        print("-" * 80)


if __name__ == "__main__":
    main()
