"""FAISS-based vector store for RAG document retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy-loaded FAISS index and metadata
_faiss_index = None
_chunks_metadata = None
_store_path = Path(__file__).parent / ".vectorstore"


def _ensure_store_dir():
    """Ensure the vector store directory exists."""
    _store_path.mkdir(exist_ok=True)


def get_or_create_index():
    """Get the FAISS index, creating if necessary."""
    global _faiss_index, _chunks_metadata
    
    if _faiss_index is not None:
        return _faiss_index, _chunks_metadata
    
    try:
        import faiss
        import numpy as np
    except ImportError as e:
        logger.error("FAISS not installed: %s", str(e))
        raise ImportError(
            "faiss-cpu or faiss-gpu is required. Install with: pip install faiss-cpu"
        ) from e
    
    _ensure_store_dir()
    
    index_path = _store_path / "index.idx"
    metadata_path = _store_path / "metadata.json"
    
    # Try to load existing index
    if index_path.exists() and metadata_path.exists():
        try:
            logger.info("Loading existing vector store from %s", _store_path)
            _faiss_index = faiss.read_index(str(index_path))
            with open(metadata_path, "r", encoding="utf-8") as f:
                _chunks_metadata = json.load(f)
            logger.info("Loaded %d chunks from vector store", len(_chunks_metadata))
            return _faiss_index, _chunks_metadata
        except Exception as e:
            logger.warning("Failed to load existing index: %s", str(e))
    
    # Create new empty index with 384 dimensions (all-MiniLM-L6-v2)
    logger.info("Creating new FAISS index with 384 dimensions")
    _faiss_index = faiss.IndexFlatL2(384)
    _chunks_metadata = []
    
    return _faiss_index, _chunks_metadata


def add_chunks(chunks: list[dict]) -> int:
    """
    Add document chunks to the vector store.
    
    Args:
        chunks: List of dicts with 'id', 'content', 'domain', 'embedding'
        
    Returns:
        Number of chunks added
    """
    import numpy as np
    
    if not chunks:
        logger.warning("No chunks to add")
        return 0
    
    index, metadata = get_or_create_index()
    
    # Extract embeddings and metadata
    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    
    logger.info("Adding %d chunks to vector store", len(chunks))
    
    # Add to FAISS index
    index.add(embeddings)
    
    # Store metadata
    for chunk in chunks:
        metadata.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "domain": chunk.get("domain", "general"),
            "index": len(metadata),
        })
    
    # Persist to disk
    _save_index()
    
    logger.info("Successfully added %d chunks", len(chunks))
    return len(chunks)


def search(embedding: list[float], k: int = 5) -> list[dict]:
    """
    Search for top-k similar chunks.
    
    Args:
        embedding: Query embedding vector
        k: Number of results to return
        
    Returns:
        List of chunk dicts with scores
    """
    import numpy as np
    
    index, metadata = get_or_create_index()
    
    if len(metadata) == 0:
        logger.warning("Vector store is empty")
        return []
    
    # Convert to numpy array and search
    query_embedding = np.array([embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding, min(k, len(metadata)))
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk = metadata[int(idx)]
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "domain": chunk["domain"],
                "similarity_score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity
            })
    
    logger.debug("Found %d similar chunks for query", len(results))
    return results


def clear_store():
    """Clear the vector store."""
    global _faiss_index, _chunks_metadata
    
    logger.info("Clearing vector store")
    _faiss_index = None
    _chunks_metadata = None
    
    # Delete persisted files
    import shutil
    if _store_path.exists():
        shutil.rmtree(_store_path)
        logger.info("Deleted vector store directory")


def _save_index():
    """Persist index and metadata to disk."""
    import faiss
    
    _ensure_store_dir()
    
    try:
        index_path = _store_path / "index.idx"
        metadata_path = _store_path / "metadata.json"
        
        faiss.write_index(_faiss_index, str(index_path))
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(_chunks_metadata, f, indent=2)
        
        logger.debug("Persisted vector store to disk")
    except Exception as e:
        logger.error("Failed to save vector store: %s", str(e))
        raise


def get_store_stats() -> dict:
    """Get vector store statistics."""
    index, metadata = get_or_create_index()
    
    domains = {}
    for chunk in metadata:
        domain = chunk.get("domain", "general")
        domains[domain] = domains.get(domain, 0) + 1
    
    return {
        "total_chunks": len(metadata),
        "index_dimension": 384,
        "domains": domains,
    }
