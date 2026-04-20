"""
RAG Engine — Retrieval-Augmented Generation core module for StudyMind AI.

This module handles:
  - Document ingestion and chunking
  - Embedding generation using local sentence-transformers (all-MiniLM-L6-v2)
  - Vector storage and semantic retrieval via ChromaDB
  - Confidence scoring for retrieved context

Using a local embedding model means:
  - No API key required for embeddings (cost-effective and private)
  - Fast inference on CPU
  - Reproducible results
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 600          # Target characters per chunk
CHUNK_OVERLAP = 100       # Characters of overlap between consecutive chunks
TOP_K = 4                 # Number of chunks to retrieve per query
MIN_CONFIDENCE = 0.25     # Minimum cosine similarity to include a chunk
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Lightweight, fast local model


# ---------------------------------------------------------------------------
# RAGEngine class
# ---------------------------------------------------------------------------
class RAGEngine:
    """
    Manages document ingestion, vector storage, and semantic retrieval.

    Uses a local sentence-transformers model for embeddings (no API key needed)
    and ChromaDB for in-memory vector storage.

    Attributes
    ----------
    chroma_client : chromadb.Client
        In-memory ChromaDB client for vector storage.
    collection : chromadb.Collection
        ChromaDB collection holding document embeddings.
    sources : List[str]
        List of source document paths that have been ingested.
    """

    def __init__(self, collection_name: str = "studymind_kb"):
        self.chroma_client = chromadb.Client()

        # Use local sentence-transformers model for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        self.sources: List[str] = []
        logger.info(
            "RAGEngine initialised with collection '%s' using embedding model '%s'.",
            collection_name, EMBEDDING_MODEL,
        )

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single text file into the vector store.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to a .txt or .md file.

        Returns
        -------
        int
            Number of chunks added to the collection.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        chunks = self._chunk_text(text, source=str(path))
        if not chunks:
            logger.warning("No chunks produced from %s.", file_path)
            return 0

        ids = [self._chunk_id(c["text"], str(path), i) for i, c in enumerate(chunks)]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

        # Upsert to avoid duplicates on re-ingestion
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        if str(path) not in self.sources:
            self.sources.append(str(path))

        logger.info("Ingested %d chunks from '%s'.", len(chunks), path.name)
        return len(chunks)

    def ingest_directory(self, directory: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> Dict[str, int]:
        """
        Ingest all matching files from a directory.

        Parameters
        ----------
        directory : str
            Path to the directory containing study notes.
        extensions : tuple
            File extensions to include.

        Returns
        -------
        dict
            Mapping of filename → number of chunks ingested.
        """
        results: Dict[str, int] = {}
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files = [f for f in dir_path.rglob("*") if f.suffix in extensions]
        if not files:
            logger.warning("No files found in '%s' with extensions %s.", directory, extensions)
            return results

        for file_path in sorted(files):
            try:
                count = self.ingest_file(str(file_path))
                results[file_path.name] = count
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to ingest '%s': %s", file_path.name, exc)
                results[file_path.name] = 0

        logger.info(
            "Directory ingestion complete. Files: %d, Total chunks: %d.",
            len(results), sum(results.values()),
        )
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Parameters
        ----------
        query : str
            The user's question or search query.
        top_k : int
            Maximum number of chunks to return.

        Returns
        -------
        list of dict
            Each dict contains 'text', 'source', 'chunk_index', and 'confidence'.
        """
        if self.collection.count() == 0:
            logger.warning("Knowledge base is empty. Ingest documents first.")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns squared L2 distance for SentenceTransformer embeddings.
            # Convert to a [0, 1] confidence score: confidence = 1 / (1 + distance)
            confidence = round(1.0 / (1.0 + dist), 4)
            if confidence < MIN_CONFIDENCE:
                continue
            chunks.append({
                "text": doc,
                "source": Path(meta["source"]).name,
                "chunk_index": meta["chunk_index"],
                "confidence": confidence,
            })

        logger.debug("Retrieved %d chunks for query: '%s'.", len(chunks), query[:60])
        return chunks

    def get_context_string(self, query: str, top_k: int = TOP_K) -> Tuple[str, float]:
        """
        Build a formatted context string from retrieved chunks.

        Returns
        -------
        tuple
            (context_string, average_confidence)
        """
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return "", 0.0

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['source']} | Confidence: {chunk['confidence']:.2f}]\n"
                f"{chunk['text']}"
            )

        avg_confidence = sum(c["confidence"] for c in chunks) / len(chunks)
        return "\n\n---\n\n".join(parts), round(avg_confidence, 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_index": idx,
                })
                idx += 1
            start = end - CHUNK_OVERLAP
        return chunks

    @staticmethod
    def _chunk_id(text: str, source: str, index: int) -> str:
        """Generate a deterministic ID for a chunk."""
        content = f"{source}::{index}::{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    @property
    def document_count(self) -> int:
        """Number of chunks currently in the knowledge base."""
        return self.collection.count()
