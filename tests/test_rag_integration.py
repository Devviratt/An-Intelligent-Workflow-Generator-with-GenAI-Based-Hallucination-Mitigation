"""Integration tests for RAG (Retrieval-Augmented Generation) system."""

from __future__ import annotations

import pytest
from rag.embedding import generate_embedding
from rag.retriever import Retriever
from rag.vector_store import (
    add_chunks,
    clear_store,
    get_store_stats,
    search,
)


class TestEmbedding:
    """Test text embedding generation."""
    
    def test_generate_embedding_basic(self) -> None:
        """Test basic embedding generation."""
        query = "Create a payment processing workflow"
        embedding = generate_embedding(query)
        
        assert embedding is not None
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        assert all(isinstance(v, float) for v in embedding)
    
    def test_generate_embedding_different_texts(self) -> None:
        """Test that different texts produce different embeddings."""
        embedding1 = generate_embedding("payment workflow")
        embedding2 = generate_embedding("user registration")
        embedding3 = generate_embedding("payment workflow")  # Same as embedding1
        
        # Different texts should produce different embeddings
        assert embedding1 != embedding2
        # Same text should produce same embedding
        assert embedding1 == embedding3
    
    def test_generate_embedding_empty_string(self) -> None:
        """Test embedding generation with empty string."""
        with pytest.raises(ValueError, match="non-empty string"):
            generate_embedding("")
    
    def test_generate_embedding_invalid_input(self) -> None:
        """Test embedding generation with invalid input."""
        with pytest.raises(ValueError, match="non-empty string"):
            generate_embedding(None)  # type: ignore


class TestVectorStore:
    """Test FAISS vector store operations."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clear store before and after each test."""
        clear_store()
        yield
        clear_store()
    
    def test_add_chunks(self) -> None:
        """Test adding chunks to vector store."""
        chunks = [
            {
                "id": "chunk_1",
                "content": "Step 1: Validate payment",
                "domain": "payment",
                "embedding": generate_embedding("Validate payment"),
            },
            {
                "id": "chunk_2",
                "content": "Step 2: Process transaction",
                "domain": "payment",
                "embedding": generate_embedding("Process transaction"),
            },
        ]
        
        count = add_chunks(chunks)
        assert count == 2
        
        # Verify stats
        stats = get_store_stats()
        assert stats["total_chunks"] == 2
        assert stats["domains"]["payment"] == 2
    
    def test_search_basic(self) -> None:
        """Test basic vector search."""
        # Add sample chunks
        chunks = [
            {
                "id": "chunk_payment",
                "content": "Process payment transaction",
                "domain": "payment",
                "embedding": generate_embedding("payment"),
            },
            {
                "id": "chunk_user",
                "content": "Register new user account",
                "domain": "user",
                "embedding": generate_embedding("user registration"),
            },
        ]
        add_chunks(chunks)
        
        # Search for payment-related content
        query_embedding = generate_embedding("payment workflow")
        results = search(query_embedding, k=2)
        
        assert len(results) > 0
        assert all("id" in r and "content" in r for r in results)
        assert "similarity_score" in results[0]
    
    def test_search_empty_store(self) -> None:
        """Test searching empty vector store."""
        query_embedding = generate_embedding("test query")
        results = search(query_embedding, k=5)
        
        assert results == []
    
    def test_search_k_parameter(self) -> None:
        """Test k parameter limits results."""
        # Add more chunks than k
        chunks = [
            {
                "id": f"chunk_{i}",
                "content": f"Content {i}",
                "domain": "general",
                "embedding": generate_embedding(f"Content {i}"),
            }
            for i in range(10)
        ]
        add_chunks(chunks)
        
        query_embedding = generate_embedding("test")
        results_k2 = search(query_embedding, k=2)
        results_k5 = search(query_embedding, k=5)
        
        assert len(results_k2) == 2
        assert len(results_k5) == 5
    
    def test_get_store_stats(self) -> None:
        """Test vector store statistics."""
        chunks = [
            {
                "id": f"chunk_{d}_{i}",
                "content": f"Content for {d}",
                "domain": d,
                "embedding": generate_embedding(f"Content for {d}"),
            }
            for d in ["payment", "user", "order"]
            for i in range(3)
        ]
        add_chunks(chunks)
        
        stats = get_store_stats()
        assert stats["total_chunks"] == 9
        assert stats["index_dimension"] == 384
        assert stats["domains"]["payment"] == 3
        assert stats["domains"]["user"] == 3
        assert stats["domains"]["order"] == 3


class TestRetriever:
    """Test Retriever class."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clear store before and after each test."""
        clear_store()
        yield
        clear_store()
    
    def test_retriever_initialization(self) -> None:
        """Test retriever initialization."""
        retriever = Retriever()
        assert retriever is not None
    
    def test_retriever_retrieve(self) -> None:
        """Test retriever.retrieve method."""
        # Setup
        chunks = [
            {
                "id": "pay_1",
                "content": "Validate customer payment information",
                "domain": "payment",
                "embedding": generate_embedding("payment validation"),
            },
            {
                "id": "pay_2",
                "content": "Process the payment transaction",
                "domain": "payment",
                "embedding": generate_embedding("payment processing"),
            },
        ]
        add_chunks(chunks)
        
        # Test
        retriever = Retriever()
        query_embedding = generate_embedding("payment verification")
        results = retriever.retrieve(query_embedding=query_embedding, k=2)
        
        assert len(results) > 0
        assert all("id" in r and "content" in r for r in results)
    
    def test_retriever_empty_store(self) -> None:
        """Test retriever with empty store."""
        retriever = Retriever()
        query_embedding = generate_embedding("test")
        results = retriever.retrieve(query_embedding=query_embedding, k=5)
        
        assert results == []


class TestEndToEnd:
    """End-to-end RAG pipeline tests."""
    
    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clear store before and after each test."""
        clear_store()
        yield
        clear_store()
    
    def test_rag_pipeline(self) -> None:
        """Test complete RAG pipeline: embed → store → retrieve."""
        # Step 1: Create sample documents
        documents = [
            ("Verify customer identity via email", "user"),
            ("Send confirmation email to customer", "user"),
            ("Process payment through gateway", "payment"),
            ("Create order in database", "order"),
            ("Ship item to customer address", "order"),
        ]
        
        # Step 2: Embed and store
        chunks = [
            {
                "id": f"doc_{i}",
                "content": content,
                "domain": domain,
                "embedding": generate_embedding(content),
            }
            for i, (content, domain) in enumerate(documents)
        ]
        add_chunks(chunks)
        stats = get_store_stats()
        assert stats["total_chunks"] == 5
        
        # Step 3: Generate query and retrieve
        query = "How do we verify users?"
        query_embedding = generate_embedding(query)
        
        retriever = Retriever()
        results = retriever.retrieve(query_embedding=query_embedding, k=3)
        
        # Step 4: Verify results
        assert len(results) == 3
        assert results[0]["domain"] in ["user", "order", "payment"]
        assert all("similarity_score" in r for r in results)
        
        # Higher similarity scores should be first
        assert (results[0]["similarity_score"] >= 
                results[1]["similarity_score"] >= 
                results[2]["similarity_score"])
    
    def test_domain_specific_retrieval(self) -> None:
        """Test retrieval across different domains."""
        # Add domain-specific chunks
        domains_data = {
            "payment": [
                "Validate payment details",
                "Process payment transaction",
                "Confirm payment receipt",
            ],
            "user": [
                "Create user account",
                "Verify user email",
                "Update user profile",
            ],
            "order": [
                "Create purchase order",
                "Update order status",
                "Ship order to customer",
            ],
        }
        
        chunks = []
        for domain, contents in domains_data.items():
            for i, content in enumerate(contents):
                chunks.append({
                    "id": f"{domain}_{i}",
                    "content": content,
                    "domain": domain,
                    "embedding": generate_embedding(content),
                })
        
        add_chunks(chunks)
        
        # Query for payment-related content
        retriever = Retriever()
        query_embedding = generate_embedding("payment processing")
        results = retriever.retrieve(query_embedding=query_embedding, k=3)
        
        # Should retrieve payment-related chunks
        assert len(results) == 3
        assert any(r["domain"] == "payment" for r in results)
