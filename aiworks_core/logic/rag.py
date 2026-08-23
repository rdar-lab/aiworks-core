"""RAG (Retrieval Augmented Generation) module for Ai-Works Core.

Provides vector database functionality for semantic search over knowledge base files.
"""

import logging
import threading
from typing import Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool as langchain_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_KB_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
_KB_CHUNK_SIZE = 1000
_KB_CHUNK_OVERLAP = 200
_KB_MAX_DISTANCE = 0.9
_KB_TOP_K = 5
_KB_BUILD_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Embedding model singleton — loaded once and reused across sessions.
# The model is ~90 MB; loading it on first use takes a few seconds.
# ---------------------------------------------------------------------------

_kb_embeddings: HuggingFaceEmbeddings | None = None
_kb_embeddings_lock = threading.Lock()


def _get_kb_embeddings() -> HuggingFaceEmbeddings:
    global _kb_embeddings
    if _kb_embeddings is None:
        with _kb_embeddings_lock:
            if _kb_embeddings is None:
                logger.info("Loading KB embedding model: %s", _KB_EMBEDDING_MODEL_NAME)
                _kb_embeddings = HuggingFaceEmbeddings(
                    model_name=_KB_EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True},
                )
                logger.info("KB embedding model loaded")
    return _kb_embeddings


def build_kb_rag_tools(kb_files: List[Dict]) -> List[BaseTool]:
    """Build a semantic vector-search RAG tool over the provided knowledge base files.

    Uses an in-memory ChromaDB vector store with HuggingFace sentence-transformer
    embeddings (all-MiniLM-L6-v2). Documents are split into overlapping chunks,
    embedded, and stored in a per-session Chroma collection. The returned tool
    performs cosine-similarity search at query time.

    Args:
        kb_files: List of dicts with 'name' and 'content' keys.

    Returns:
        A list containing a single LangChain ``BaseTool`` for KB retrieval,
        or an empty list if no files are provided or all files are empty.
    """
    if not kb_files:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_KB_CHUNK_SIZE,
        chunk_overlap=_KB_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: List[str] = []
    metadatas: List[Dict] = []
    for file_data in kb_files:
        name = file_data['name']
        content = file_data.get('content', '')
        if not content:
            logger.warning("build_kb_rag_tools | skipping empty file: %s", name)
            continue
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({'file_name': name, 'chunk_index': i, 'total_chunks': len(chunks)})

    if not texts:
        return []

    client = chromadb.Client(ChromaSettings(is_persistent=False, anonymized_telemetry=False))
    vectorstore = Chroma(
        client=client,
        collection_name='kb_session',
        embedding_function=_get_kb_embeddings(),
    )

    if len(texts) < _KB_BUILD_BATCH_SIZE:
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    else:
        for i in range(0, len(texts), _KB_BUILD_BATCH_SIZE):
            batch_texts = texts[i:i + _KB_BUILD_BATCH_SIZE]
            batch_metadatas = metadatas[i:i + _KB_BUILD_BATCH_SIZE]
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)

    logger.info("build_kb_rag_tools | indexed %d chunks from %d files", len(texts), len(kb_files))

    def _search_kb(query: str) -> str:
        logger.info(f"Searching KB for query: {query}")
        results = vectorstore.similarity_search_with_score(query=query, k=_KB_TOP_K)
        passages = [
            f"[From: {doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
            for doc, score in results
            if score <= _KB_MAX_DISTANCE
        ]
        logger.info(f"KB search found {len(passages)} relevant passages (out of {len(results)} results)")
        if not passages:
            return "No relevant content found in the knowledge base for this query."
        return '\n\n---\n\n'.join(passages)

    @langchain_tool
    def search_internal_knowledge_base(query: str) -> str:
        """Search the internal knowledge base documents (private, non-internet data uploaded by the user)
        for information relevant to the given query. Uses semantic similarity search to find the most
        relevant passages from the internal knowledge base files."""
        return _search_kb(query)

    return [search_internal_knowledge_base]
