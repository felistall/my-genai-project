# src/7_document_loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

PDF_PATH = "data/rag_explained.pdf"   # Use the PDF we generated

def load_and_split(pdf_path):
    # 1) Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # 2) Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    return chunks


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
    else:
        chunks = load_and_split(PDF_PATH)

        print("\nFirst chunk preview:")
        print(chunks[0].page_content)
