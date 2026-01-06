import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5