import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = "2025-01-01-preview"

# Azure deployment names
LLM_DEPLOYMENT = "gpt-4.1"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5   # Initial retrieval count
TOP_N = 3   # After re-ranking

# Paths
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
