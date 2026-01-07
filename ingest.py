import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL
)


def ingest_user_document(file_path, vectorstore):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        print("Unsupported file format.")
        return
    
    docs = loader.load()
    docs_with_metadata = []
    for doc in docs:
        doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": file_path,
                "uploaded_by": "user",
                **getattr(doc, "metadata", {})
            }
        )
        docs_with_metadata.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs_with_metadata)
    vectorstore.add_documents(chunks)
    print("Document ingested successfully!")


def load_documents(data_dir="data"):
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return []
    
    documents = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename  # Add filename as source
        documents.extend(docs)
    
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def store_embeddings(chunks):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vectorstore


def batch_ingest(data_dir="data"):
    print(" Loading documents from directory 📄")
    documents = load_documents(data_dir)
    if not documents:
        print("No documents found. Skipping ingestion.")
        return None
    
    print(f"Loaded {len(documents)} documents")
    
    print(" Splitting into chunks ✂️")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print(" Generating embeddings and storing in Chroma 🧠")
    vectorstore = store_embeddings(chunks)
    print(" Batch ingestion complete ✅")
    return vectorstore