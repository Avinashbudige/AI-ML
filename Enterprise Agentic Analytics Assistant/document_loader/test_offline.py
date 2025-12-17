#!/usr/bin/env python3
"""
Simple offline test script for the Document Loader module (no network required).
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the repository root to the Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

ea3_path = repo_root / "Enterprise Agentic Analytics Assistant"
sys.path.insert(0, str(ea3_path))

# Import document_loader components
from document_loader import (
    MarkdownLoader,
    RecursiveChunker,
    SemanticChunker,
    config
)


def test_markdown_loading():
    """Test loading markdown documents."""
    print("=" * 60)
    print("Test 1: Markdown Loading")
    print("=" * 60)
    
    # Create a sample markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        sample_md = f.name
        content = """# Introduction to AI

Artificial Intelligence is transforming the world.

## Machine Learning

Machine learning is a subset of AI that learns from data.

### Types of ML
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

## Deep Learning

Deep learning uses neural networks with multiple layers.
"""
        f.write(content)
    
    try:
        # Load the markdown file
        loader = MarkdownLoader(sample_md)
        documents = loader.load()
        
        print(f"✓ Loaded {len(documents)} document(s)")
        print(f"✓ Content length: {len(documents[0]['content'])} characters")
        print(f"✓ Headers found: {len(documents[0]['metadata']['headers'])}")
        print(f"✓ File type: {documents[0]['metadata']['file_type']}")
        
        # Load by sections
        sections = loader.load_by_sections()
        print(f"✓ Split into {len(sections)} sections")
        
        for i, section in enumerate(sections):
            header = section['metadata'].get('header', 'N/A')
            print(f"  Section {i+1}: {header}")
        
        print()
        return documents
        
    finally:
        # Clean up
        if os.path.exists(sample_md):
            os.unlink(sample_md)


def test_recursive_chunking(documents):
    """Test recursive chunking."""
    print("=" * 60)
    print("Test 2: Recursive Chunking")
    print("=" * 60)
    
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(documents)
    
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} document(s)")
    print(f"✓ Chunk size setting: {chunker.chunk_size}")
    print(f"✓ Chunk overlap setting: {chunker.chunk_overlap}")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n  Chunk {i+1}:")
        print(f"    Size: {chunk['metadata']['chunk_size']} characters")
        print(f"    Method: {chunk['metadata']['chunk_method']}")
        print(f"    Index: {chunk['metadata']['chunk_index']}")
        print(f"    Preview: {chunk['content'][:80].replace(chr(10), ' ')}...")
    
    print()
    return chunks


def test_semantic_chunking(documents):
    """Test semantic chunking."""
    print("=" * 60)
    print("Test 3: Semantic Chunking")
    print("=" * 60)
    
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk(documents)
    
    print(f"✓ Created {len(chunks)} semantic chunks from {len(documents)} document(s)")
    print(f"✓ Chunk size setting: {chunker.chunk_size}")
    print(f"✓ Chunk overlap setting: {chunker.chunk_overlap}")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n  Chunk {i+1}:")
        print(f"    Size: {chunk['metadata']['chunk_size']} characters")
        print(f"    Method: {chunk['metadata']['chunk_method']}")
        print(f"    Preview: {chunk['content'][:80].replace(chr(10), ' ')}...")
    
    print()
    return chunks


def test_configuration():
    """Test configuration settings."""
    print("=" * 60)
    print("Test 4: Configuration")
    print("=" * 60)
    
    cfg = config.get_config()
    
    print(f"✓ Chunk size: {cfg['chunk_size']}")
    print(f"✓ Chunk overlap: {cfg['chunk_overlap']}")
    print(f"✓ Default embedding model: {cfg['default_embedding_model']}")
    print(f"✓ OpenAI embedding model: {cfg['openai_embedding_model']}")
    print(f"✓ BGE embedding model: {cfg['bge_embedding_model']}")
    print(f"✓ Default vector store: {cfg['default_vector_store']}")
    print(f"✓ FAISS index type: {cfg['faiss_index_type']}")
    print(f"✓ Semantic similarity threshold: {cfg['semantic_similarity_threshold']}")
    print()


def test_file_not_found():
    """Test error handling for non-existent file."""
    print("=" * 60)
    print("Test 5: Error Handling")
    print("=" * 60)
    
    try:
        loader = MarkdownLoader("/nonexistent/file.md")
        print("✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ FileNotFoundError correctly raised: {str(e)[:60]}...")
    
    # Create a dummy file with wrong extension
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        dummy_file = f.name
        f.write("test content")
    
    try:
        from document_loader import PDFLoader
        loader = PDFLoader(dummy_file)  # Wrong extension
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ ValueError correctly raised for wrong extension")
    finally:
        if os.path.exists(dummy_file):
            os.unlink(dummy_file)
    
    print()


def test_module_imports():
    """Test that all module components can be imported."""
    print("=" * 60)
    print("Test 6: Module Imports")
    print("=" * 60)
    
    try:
        from document_loader import (
            BaseDocumentLoader,
            PDFLoader,
            MarkdownLoader,
            RecursiveChunker,
            SemanticChunker,
            EmbeddingGenerator,
            VectorStorePreparation
        )
        
        print("✓ BaseDocumentLoader imported")
        print("✓ PDFLoader imported")
        print("✓ MarkdownLoader imported")
        print("✓ RecursiveChunker imported")
        print("✓ SemanticChunker imported")
        print("✓ EmbeddingGenerator imported")
        print("✓ VectorStorePreparation imported")
        print()
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Document Loader Module - Offline Test Suite")
    print("=" * 60 + "\n")
    
    # Run tests
    test_module_imports()
    test_configuration()
    documents = test_markdown_loading()
    chunks_recursive = test_recursive_chunking(documents)
    chunks_semantic = test_semantic_chunking(documents)
    test_file_not_found()
    
    print("=" * 60)
    print("All offline tests completed successfully!")
    print("=" * 60)
    print("\nNote: Online tests (embeddings, vector stores) require")
    print("network access and are skipped in this offline test.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
