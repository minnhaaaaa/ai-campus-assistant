import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from ingest import ingest_user_document, batch_ingest

from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    OPENAI_API_KEY
)

# -----------------------------
# Load Vector Store (for querying)
# -----------------------------
def load_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore

# -----------------------------
# Build Prompt
# -----------------------------
def build_prompt(context, question):
    system_prompt = (
        "You are a knowledge-aware AI assistant.\n"
        "Answer the user's question strictly using the provided context.\n"
        "If the answer is not present in the context, say:\n"
        "\"I don't know based on the provided documents.\""
    )
    
    user_prompt = f"""
Context:
{context}

Question:
{question}
"""
    
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

# -----------------------------
# Answer Query
# -----------------------------
def answer_query(llm, vectorstore, query):
    docs = vectorstore.similarity_search(query, k=TOP_K)
    
    if not docs:
        return "I don't know based on the provided documents."
    
    context = "\n\n".join(
        [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
         for doc in docs]
    )
    
    messages = build_prompt(context, query)
    response = llm.invoke(messages)
    
    return response.content

# -----------------------------
# CLI Loop
# -----------------------------
def main():
    print("📚 Knowledge-Aware AI Assistant (CLI)")
    print("Commands: 'exit' to quit, 'upload <path>' for single file, 'ingest <dir>' for batch.\n")
    
    vectorstore = load_vector_store()
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0,openai_api_key=OPENAI_API_KEY)
    
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break
        elif query.startswith("upload "):
            path = query.replace("upload ", "").strip()
            ingest_user_document(path, vectorstore)  # Call from ingest.py
            # Reload vectorstore to reflect changes
            vectorstore = load_vector_store()
            continue
        elif query.startswith("ingest "):
            data_dir = query.replace("ingest ", "").strip() or "data"
            new_vectorstore = batch_ingest(data_dir)  # Call from ingest.py
            if new_vectorstore:
                vectorstore = new_vectorstore  # Update to the new batch-loaded store
            continue
        
        answer = answer_query(llm, vectorstore, query)
        print(f"\nAI: {answer}\n")

# -----------------------------
if __name__ == "__main__":
    main()