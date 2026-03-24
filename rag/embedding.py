"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid initialization overhead
_embedding_model = None


def _get_embedding_model():
    """Load the embedding model lazily on first use."""
    global _embedding_model
    
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            
            # Using all-MiniLM-L6-v2: fast, lightweight (22M params), 384-dim embeddings
            # Good balance between speed and quality for RAG tasks
            model_name = "all-MiniLM-L6-v2"
            logger.info("Loading embedding model: %s", model_name)
            _embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError as e:
            logger.error("sentence-transformers not installed: %s", str(e))
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            ) from e
    
    return _embedding_model


def generate_embedding(query: str) -> list[float]:
    """
    Generate embeddings for a query string using sentence-transformers.
    
    Args:
        query: Text query to embed
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        ValueError: If query is empty or None
        ImportError: If sentence-transformers is not installed
    """
    if not query or not isinstance(query, str):
        logger.error("Invalid query provided: %s", query)
        raise ValueError("Query must be a non-empty string")
    
    try:
        model = _get_embedding_model()
        
        logger.debug("Generating embedding for query: %s", query[:100])
        
        # Generate embeddings (returns numpy array)
        embeddings = model.encode([query], convert_to_tensor=False)
        
        # Convert to list and return first (and only) embedding
        embedding_list = embeddings[0].tolist()
        
        logger.debug(
            "Embedding generated with %d dimensions",
            len(embedding_list),
        )
        
        return embedding_list
        
    except Exception as e:
        logger.error("Failed to generate embedding: %s", str(e))
        raise
