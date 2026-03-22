import os
from typing import List, Optional

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_API_VERSION, EMBEDDING_DEPLOYMENT, FAISS_INDEX_DIR


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Get the Azure OpenAI embedding model."""
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
    )


def create_faiss_index(chunks: List[Document]) -> FAISS:
    """Create a FAISS index from document chunks and persist it."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"FAISS index created with {len(chunks)} vectors, saved to {FAISS_INDEX_DIR}")
    return vectorstore


def load_faiss_index() -> Optional[FAISS]:
    """Load an existing FAISS index from disk."""
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_file):
        print("No existing FAISS index found.")
        return None
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("FAISS index loaded from disk.")
    return vectorstore


def add_to_faiss_index(vectorstore: FAISS, chunks: List[Document]) -> FAISS:
    """Add new chunks to an existing FAISS index."""
    vectorstore.add_documents(chunks)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"Added {len(chunks)} chunks to FAISS index.")
    return vectorstore
