from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import TOP_K


def retrieve_documents(vectorstore: FAISS, query: str, k: int = TOP_K) -> List[Document]:
    """Retrieve top-k relevant documents for a query using similarity search."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(query)
    return docs


def retrieve_with_scores(vectorstore: FAISS, query: str, k: int = TOP_K) -> List[tuple]:
    """Retrieve documents with relevance scores (lower = more similar for L2)."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results
