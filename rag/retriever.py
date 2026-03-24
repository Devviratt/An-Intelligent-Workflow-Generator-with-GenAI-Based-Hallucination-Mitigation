"""Vector retrieval using FAISS for RAG."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant chunks from a vector store.
    
    Tries to use FAISS first, falls back to mock store if FAISS is unavailable.
    """
    
    def __init__(self):
        """Initialize the retriever."""
        logger.info("Initializing Retriever")
        self.use_faiss = True
        
        # Check if FAISS is available
        try:
            import faiss  # noqa: F401
        except ImportError:
            logger.warning("FAISS not available, using mock store")
            self.use_faiss = False
    
    def retrieve(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """
        Retrieve top-k relevant chunks using semantic search.
        
        Args:
            query_embedding: The embedded query vector (384 dimensions)
            k: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with 'id', 'content', 'domain', 'similarity_score'
        """
        logger.info("Retrieving top %d chunks from %s", k, 
                   "FAISS" if self.use_faiss else "mock store")
        
        try:
            if self.use_faiss:
                from rag.vector_store import search
                return search(query_embedding, k=k)
            else:
                from rag.mock_store import search_mock
                return search_mock(query_embedding, k=k)
            
        except Exception as e:
            logger.error("Retrieval failed: %s", str(e))
            return []
