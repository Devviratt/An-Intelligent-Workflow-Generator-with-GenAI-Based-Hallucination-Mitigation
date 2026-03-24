"""
Populate FAISS vector store with workflow chunks from domain datasets.

This script reads JSON domain datasets and extracts workflow steps,
then embeds and stores them in FAISS for RAG-based retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag.embedding import generate_embedding
from rag.vector_store import add_chunks, clear_store, get_store_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_datasets(datasets_dir: Path) -> dict[str, dict]:
    """
    Load all JSON domain datasets from directory.
    
    Args:
        datasets_dir: Path to datasets directory
        
    Returns:
        Dictionary mapping domain names to dataset dicts
    """
    datasets = {}
    
    if not datasets_dir.exists():
        logger.warning("Datasets directory does not exist: %s", datasets_dir)
        return datasets
    
    for json_file in datasets_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
                domain_name = json_file.stem
                datasets[domain_name] = dataset
                logger.info("Loaded dataset: %s (%s)", domain_name, json_file.name)
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", json_file.name, str(e))
    
    logger.info("Loaded %d datasets", len(datasets))
    return datasets


def extract_chunks_from_dataset(
    domain_name: str,
    dataset: dict,
) -> list[dict]:
    """
    Extract workflow step chunks from a domain dataset.
    
    Args:
        domain_name: Domain identifier
        dataset: Domain dataset dictionary
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    chunks = []
    
    # Extract description as first chunk
    if "description" in dataset:
        chunks.append({
            "id": f"{domain_name}_description",
            "content": f"{domain_name}: {dataset['description']}",
            "domain": domain_name,
        })
    
    # Extract steps
    if "steps" in dataset and isinstance(dataset["steps"], list):
        for step in dataset["steps"]:
            if isinstance(step, dict) and "label" in step:
                step_id = step.get("id", "unknown")
                step_label = step["label"]
                step_desc = step.get("description", "")
                
                # Create content combining label and description
                content = f"{step_label}: {step_desc}" if step_desc else step_label
                
                chunks.append({
                    "id": f"{domain_name}_{step_id}",
                    "content": content,
                    "domain": domain_name,
                })
    
    # Extract transitions
    if "transitions" in dataset and isinstance(dataset["transitions"], list):
        for i, transition in enumerate(dataset["transitions"]):
            if isinstance(transition, dict):
                from_step = transition.get("from_step", "?")
                to_step = transition.get("to_step", "?")
                condition = transition.get("condition", "")
                
                content = f"Transition: {from_step} → {to_step}"
                if condition:
                    content += f" [{condition}]"
                
                chunks.append({
                    "id": f"{domain_name}_transition_{i}",
                    "content": content,
                    "domain": domain_name,
                })
    
    logger.info("Extracted %d chunks from %s", len(chunks), domain_name)
    return chunks


def populate_vector_store(
    datasets_dir: Path | None = None,
    clear_first: bool = True,
    use_faiss: bool = True,
) -> int:
    """
    Populate the vector store with dataset chunks.
    
    Can use either FAISS (full vector search) or mock store (basic demo).
    
    Args:
        datasets_dir: Path to datasets directory (defaults to project datasets/)
        clear_first: Whether to clear existing store first
        use_faiss: Whether to use FAISS (True) or mock store (False)
        
    Returns:
        Total number of chunks stored
    """
    if datasets_dir is None:
        # Default to project datasets directory
        datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
    
    logger.info(
        "Populating vector store from %s (using %s)",
        datasets_dir,
        "FAISS" if use_faiss else "mock store",
    )
    
    if clear_first:
        logger.info("Clearing existing vector store")
        if use_faiss:
            from rag.vector_store import clear_store
            clear_store()
        else:
            from rag.mock_store import save_mock_store
            save_mock_store([])
    
    # Load all datasets
    datasets = load_datasets(datasets_dir)
    if not datasets:
        logger.warning("No datasets found to populate")
        return 0
    
    # Extract and embed chunks
    all_chunks = []
    for domain_name, dataset in datasets.items():
        try:
            chunks = extract_chunks_from_dataset(domain_name, dataset)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error("Failed to extract chunks from %s: %s", domain_name, str(e))
            continue
    
    logger.info("Extracted %d total chunks", len(all_chunks))
    
    # Generate embeddings for all chunks
    chunks_with_embeddings = []
    failed_count = 0
    
    for i, chunk in enumerate(all_chunks):
        try:
            embedding = generate_embedding(chunk["content"])
            chunks_with_embeddings.append({
                **chunk,
                "embedding": embedding,
            })
            
            if (i + 1) % 10 == 0:
                logger.info("Embedded %d/%d chunks", i + 1, len(all_chunks))
        except Exception as e:
            logger.warning("Failed to embed chunk %s: %s", chunk["id"], str(e))
            failed_count += 1
    
    if failed_count > 0:
        logger.warning("Failed to embed %d chunks", failed_count)
    
    # Add to vector store
    if use_faiss:
        try:
            from rag.vector_store import add_chunks, get_store_stats
            
            stored_count = add_chunks(chunks_with_embeddings)
            
            # Print statistics
            stats = get_store_stats()
            logger.info("Vector store statistics:")
            logger.info("  Total chunks: %d", stats["total_chunks"])
            logger.info("  Dimensions: %d", stats["index_dimension"])
            logger.info("  Domains: %s", stats["domains"])
        except ImportError:
            logger.warning("FAISS not available, falling back to mock store")
            from rag.mock_store import save_mock_store
            
            save_mock_store(chunks_with_embeddings)
            stored_count = len(chunks_with_embeddings)
    else:
        from rag.mock_store import save_mock_store
        
        save_mock_store(chunks_with_embeddings)
        stored_count = len(chunks_with_embeddings)
    
    return stored_count


if __name__ == "__main__":
    # Run population when script is executed directly
    import sys
    
    try:
        datasets_path = None
        use_faiss = True
        
        if len(sys.argv) > 1:
            datasets_path = Path(sys.argv[1])
        
        if "--mock" in sys.argv:
            use_faiss = False
            logger.info("Using mock store (FAISS disabled)")
        
        count = populate_vector_store(datasets_path, use_faiss=use_faiss)
        logger.info("Successfully populated vector store with %d chunks", count)
        sys.exit(0)
    except Exception as e:
        logger.error("Failed to populate vector store: %s", str(e), exc_info=True)
        sys.exit(1)
