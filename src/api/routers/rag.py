"""RAG (Retrieval-Augmented Generation) Router — query processing via vector retrieval."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.models.rag import ProcessQueryRequest, ProcessQueryResponse, RetrievedChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router.post(
    "/process_query",
    response_model=ProcessQueryResponse,
    summary="Process query through RAG pipeline",
    description=(
        "Accepts a user query, generates embeddings, and retrieves top-k relevant "
        "chunks from the vector store. Returns the original query, embedding preview, "
        "and retrieved chunks with metadata."
    ),
)
async def process_query(request: ProcessQueryRequest) -> ProcessQueryResponse:
    """
    Process a user query through the first 3 steps of RAG workflow:
    
    1. Accept user query
    2. Generate embedding using embedding model
    3. Retrieve top-k relevant chunks from FAISS vector store
    
    Args:
        request: Query processing request with query string and top_k parameter
        
    Returns:
        ProcessQueryResponse with original query, embedding preview, and chunks
        
    Raises:
        HTTPException: If embedding generation or retrieval fails
    """
    try:
        # Step 1: User query is already in request.query
        user_query = request.query
        top_k = request.top_k
        
        logger.info(
            "Processing query: %s (retrieving top %d chunks)",
            user_query[:100],  # Log first 100 chars
            top_k,
        )
        
        # Step 2: Generate embedding from query
        from rag.embedding import generate_embedding
         
        embedding = generate_embedding(user_query)
        
        if embedding is None or len(embedding) == 0:
            logger.error("Failed to generate embedding for query: %s", user_query)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding from query",
            )
        
        # Extract first 10 values for preview (or fewer if embedding is shorter)
        embedding_preview = embedding[:10]
        
        logger.debug(
            "Embedding generated with %d dimensions, preview: %s",
            len(embedding),
            embedding_preview[:5],
        )
        
        # Step 3: Retrieve top-k chunks from vector store
        from rag.retriever import Retriever
        
        retriever = Retriever()
        retrieved_results = retriever.retrieve(query_embedding=embedding, k=top_k)
        
        if retrieved_results is None:
            logger.error("Retriever returned None for query: %s", user_query)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve chunks from vector store",
            )
        
        # Transform retrieved results into RetrievedChunk models
        chunks: list[RetrievedChunk] = []
        
        for result in retrieved_results:
            try:
                chunk = RetrievedChunk(
                    id=result.get("id", "unknown"),
                    content=result.get("content", ""),
                    domain=result.get("domain", "general"),
                )
                chunks.append(chunk)
            except (KeyError, ValueError) as e:
                logger.warning(
                    "Skipping malformed chunk: %s — error: %s",
                    result,
                    str(e),
                )
                continue
        
        logger.info(
            "Successfully retrieved %d chunks for query: %s",
            len(chunks),
            user_query[:100],
        )
        
        return ProcessQueryResponse(
            user_query=user_query,
            embedding_preview=embedding_preview,
            retrieved_chunks=chunks,
        )
        
    except ImportError as e:
        logger.error("Failed to import RAG modules: %s", str(e))
        raise HTTPException(
            status_code=503,
            detail="RAG modules not available — check deployment",
        ) from e
        
    except Exception as e:
        logger.exception("Unexpected error in process_query: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}",
        ) from e
