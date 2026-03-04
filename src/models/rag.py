"""RAG (Retrieval-Augmented Generation) data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single retrieved chunk from the vector store."""

    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the chunk")
    domain: str = Field(..., description="Domain category of the chunk")


class ProcessQueryRequest(BaseModel):
    """Request to process a user query through RAG pipeline."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User query to process",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top-k chunks to retrieve",
    )


class ProcessQueryResponse(BaseModel):
    """Response from RAG query processing."""

    user_query: str = Field(..., description="Original user query")
    embedding_preview: list[float] = Field(
        ...,
        description="First 10 values of the generated embedding vector",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="List of retrieved relevant chunks from vector store",
    )
