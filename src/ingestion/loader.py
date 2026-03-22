import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
}


def load_single_document(file_path: str) -> List[Document]:
    """Load a single document and attach metadata."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        print(f"Skipping unsupported file type: {ext} ({file_path})")
        return []

    loader = loader_cls(file_path)
    docs = loader.load()

    # Enrich metadata
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["file_path"] = file_path
        doc.metadata["file_type"] = ext
        doc.metadata["doc_type"] = _infer_doc_type(filename)

    return docs


def load_documents_from_directory(directory: str) -> List[Document]:
    """Load all supported documents from a directory."""
    all_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                docs = load_single_document(file_path)
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} pages/sections from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs


def _infer_doc_type(filename: str) -> str:
    """Infer document category from filename keywords."""
    name_lower = filename.lower()
    if "annual" in name_lower:
        return "annual_report"
    elif "quarter" in name_lower or "q1" in name_lower or "q2" in name_lower or "q3" in name_lower or "q4" in name_lower:
        return "quarterly_report"
    elif "strategy" in name_lower or "strategic" in name_lower:
        return "strategy_note"
    elif "operation" in name_lower or "ops" in name_lower:
        return "operational_update"
    elif "financial" in name_lower or "finance" in name_lower:
        return "financial_report"
    else:
        return "general"
