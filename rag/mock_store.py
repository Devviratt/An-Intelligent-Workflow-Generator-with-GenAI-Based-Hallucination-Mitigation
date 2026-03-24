"""Mock vector store for testing - uses simple JSON instead of FAISS."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_store_path = Path(__file__).parent / ".vectorstore"
_mock_store = None


def _ensure_store_dir():
    """Ensure directory exists."""
    _store_path.mkdir(exist_ok=True)


def load_mock_store() -> list[dict]:
    """Load mock vector store from JSON."""
    global _mock_store
    
    if _mock_store is not None:
        return _mock_store
    
    _ensure_store_dir()
    mock_path = _store_path / "mock_chunks.json"
    
    if mock_path.exists():
        try:
            with open(mock_path, "r", encoding="utf-8") as f:
                _mock_store = json.load(f)
                logger.info("Loaded %d chunks from mock store", len(_mock_store))
                return _mock_store
        except Exception as e:
            logger.warning("Failed to load mock store: %s", str(e))
    
    _mock_store = []
    return _mock_store


def save_mock_store(chunks: list[dict]) -> None:
    """Save mock chunks to JSON."""
    global _mock_store
    
    _ensure_store_dir()
    mock_path = _store_path / "mock_chunks.json"
    
    # Remove embeddings before saving (too large)
    chunks_to_save = [
        {k: v for k, v in c.items() if k != "embedding"}
        for c in chunks
    ]
    
    with open(mock_path, "w", encoding="utf-8") as f:
        json.dump(chunks_to_save, f, indent=2)
    
    _mock_store = chunks_to_save
    logger.info("Saved %d chunks to mock store", len(chunks_to_save))


def search_mock(query_embedding: list[float], k: int = 5) -> list[dict]:
    """
    Simple semantic search using cosine similarity on mock data.
    Uses minimal similarity computation for demo purposes.
    """
    chunks = load_mock_store()
    
    if not chunks:
        logger.warning("Mock store is empty")
        return []
    
    # For mock search, just return top-k chunks in order
    # In a real system, this would use FAISS with actual embeddings
    results = []
    for i, chunk in enumerate(chunks[:k]):
        # Mock similarity score (random for now)
        similarity = 1.0 - (i * 0.1)  # Decreasing scores
        results.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "domain": chunk.get("domain", "general"),
            "similarity_score": max(0.5, similarity),  # Min 0.5
        })
    
    return results
