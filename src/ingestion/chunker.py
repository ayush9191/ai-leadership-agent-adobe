import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents using RecursiveCharacterTextSplitter.

    Splits on paragraph breaks → line breaks → sentences → words, in order,
    so natural boundaries are always preferred over mid-sentence cuts.
    Each chunk inherits the source document's metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks "
          f"(recursive, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return all_chunks
