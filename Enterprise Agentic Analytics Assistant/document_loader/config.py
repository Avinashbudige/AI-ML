"""
Configuration settings for the document loader module.
"""

import os
from typing import Dict, Any

# Chunking configuration
CHUNK_SIZE = 1000  # Default chunk size in characters
CHUNK_OVERLAP = 200  # Default overlap between chunks

# Embedding configuration
DEFAULT_EMBEDDING_MODEL = "openai"  # Options: "openai", "bge"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
BGE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Vector store configuration
DEFAULT_VECTOR_STORE = "faiss"  # Options: "faiss", "opensearch"
FAISS_INDEX_TYPE = "FlatL2"  # Options: "FlatL2", "IVFFlat", "HNSW"

# File processing settings
SUPPORTED_PDF_EXTENSIONS = [".pdf"]
SUPPORTED_MARKDOWN_EXTENSIONS = [".md", ".markdown"]

# Semantic chunking settings
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # Threshold for semantic chunking


def get_openai_api_key() -> str:
    """
    Safely retrieve OpenAI API key from environment.
    
    Returns:
        str: API key from environment variable, or empty string if not set
    """
    return os.getenv("OPENAI_API_KEY", "")


def get_config() -> Dict[str, Any]:
    """
    Returns the current configuration as a dictionary.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "bge_embedding_model": BGE_EMBEDDING_MODEL,
        "default_vector_store": DEFAULT_VECTOR_STORE,
        "faiss_index_type": FAISS_INDEX_TYPE,
        "semantic_similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
    }
