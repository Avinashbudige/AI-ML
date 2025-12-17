"""
Document Loader Module for Enterprise Agentic Analytics Assistant (EA³)

This module provides functionality to load, process, and prepare unstructured data
for use in the analytics assistant. It supports multiple document formats, chunking
strategies, and embedding generation.
"""

from .base_loader import BaseDocumentLoader
from .pdf_loader import PDFLoader
from .markdown_loader import MarkdownLoader
from .chunking import RecursiveChunker, SemanticChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStorePreparation

__version__ = "1.0.0"

__all__ = [
    "BaseDocumentLoader",
    "PDFLoader",
    "MarkdownLoader",
    "RecursiveChunker",
    "SemanticChunker",
    "EmbeddingGenerator",
    "VectorStorePreparation",
]
