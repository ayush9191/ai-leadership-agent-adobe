"""Standalone script to build/rebuild the FAISS index. Run in a subprocess."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.loader import load_documents_from_directory
from src.ingestion.chunker import chunk_documents
from src.vectorstore.store import create_faiss_index
from config import DOCUMENTS_DIR

docs = load_documents_from_directory(DOCUMENTS_DIR)
chunks = chunk_documents(docs)
vs = create_faiss_index(chunks)
print(f"INDEX_BUILT:{len(chunks)}")
