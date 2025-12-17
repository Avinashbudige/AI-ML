#!/usr/bin/env python3
"""
Simple test script for the Document Loader module.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the repository root to the Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Now import from the Enterprise Agentic Analytics Assistant package
# We need to create a proper package structure
ea3_path = repo_root / "Enterprise Agentic Analytics Assistant"
sys.path.insert(0, str(ea3_path))

# Import document_loader components
from document_loader import (
    MarkdownLoader,
    RecursiveChunker,
    SemanticChunker,
    EmbeddingGenerator,
    VectorStorePreparation
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
    
    with open(sample_md, 'w') as f:
        f.write(content)
    
    # Load the markdown file
    loader = MarkdownLoader(sample_md)
    documents = loader.load()
    
    print(f"✓ Loaded {len(documents)} document(s)")
    print(f"✓ Content length: {len(documents[0]['content'])} characters")
    print(f"✓ Headers found: {len(documents[0]['metadata']['headers'])}")
    
    # Load by sections
    sections = loader.load_by_sections()
    print(f"✓ Split into {len(sections)} sections")
    print()
    
    return documents


def test_recursive_chunking(documents):
    """Test recursive chunking."""
    print("=" * 60)
    print("Test 2: Recursive Chunking")
    print("=" * 60)
    
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(documents)
    
    print(f"✓ Created {len(chunks)} chunks")
    print(f"✓ First chunk size: {chunks[0]['metadata']['chunk_size']} characters")
    print(f"✓ Chunk method: {chunks[0]['metadata']['chunk_method']}")
    print()
    
    return chunks


def test_semantic_chunking(documents):
    """Test semantic chunking."""
    print("=" * 60)
    print("Test 3: Semantic Chunking")
    print("=" * 60)
    
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk(documents)
    
    print(f"✓ Created {len(chunks)} semantic chunks")
    print(f"✓ First chunk size: {chunks[0]['metadata']['chunk_size']} characters")
    print()
    
    return chunks


def test_bge_embeddings(chunks):
    """Test BGE embedding generation."""
    print("=" * 60)
    print("Test 4: BGE Embeddings")
    print("=" * 60)
    
    try:
        # Create embedding generator
        embedding_gen = EmbeddingGenerator(model_type="bge", device="cpu")
        
        # Generate embeddings for first few chunks
        test_chunks = chunks[:3]
        embedded_docs = embedding_gen.embed_documents(test_chunks)
        
        print(f"✓ Generated embeddings for {len(embedded_docs)} chunks")
        print(f"✓ Embedding dimension: {embedding_gen.get_embedding_dimension()}")
        print(f"✓ Embedding shape: {embedded_docs[0]['embedding'].shape}")
        print()
        
        return embedding_gen, embedded_docs
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return None, None


def test_faiss_preparation(embedding_gen, embedded_docs):
    """Test FAISS preparation."""
    print("=" * 60)
    print("Test 5: FAISS Preparation")
    print("=" * 60)
    
    if embedding_gen is None or embedded_docs is None:
        print("✗ Skipped (embeddings not available)")
        print()
        return None
    
    try:
        # Prepare for FAISS
        embedding_dim = embedding_gen.get_embedding_dimension()
        vector_prep = VectorStorePreparation(embedding_dimension=embedding_dim)
        
        faiss_data = vector_prep.prepare_for_faiss(
            documents=embedded_docs,
            index_type="FlatL2"
        )
        
        print(f"✓ Created FAISS index")
        print(f"✓ Number of vectors: {faiss_data['num_vectors']}")
        print(f"✓ Embedding dimension: {faiss_data['embedding_dimension']}")
        print(f"✓ Index type: {faiss_data['index_type']}")
        
        # Save index
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as idx_file:
            index_path = idx_file.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as meta_file:
            metadata_path = meta_file.name
        
        vector_prep.save_faiss_index(
            faiss_data=faiss_data,
            index_path=index_path,
            metadata_path=metadata_path
        )
        print()
        
        return faiss_data, vector_prep
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return None, None


def test_faiss_search(faiss_data, vector_prep, embedding_gen):
    """Test FAISS similarity search."""
    print("=" * 60)
    print("Test 6: FAISS Similarity Search")
    print("=" * 60)
    
    if faiss_data is None or vector_prep is None or embedding_gen is None:
        print("✗ Skipped (FAISS index not available)")
        print()
        return
    
    try:
        # Create a query
        query = "What is machine learning?"
        query_embedding = embedding_gen.generate_embeddings([query])[0]
        
        # Search
        results = vector_prep.search_similar(
            faiss_index=faiss_data['index'],
            query_embedding=query_embedding,
            metadata=faiss_data['metadata'],
            k=2
        )
        
        print(f"✓ Query: '{query}'")
        print(f"✓ Found {len(results)} similar documents")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.4f}, "
                  f"Content: {result['document']['content'][:50]}...")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()


def test_opensearch_preparation(embedding_gen, embedded_docs):
    """Test OpenSearch preparation."""
    print("=" * 60)
    print("Test 7: OpenSearch Preparation")
    print("=" * 60)
    
    if embedding_gen is None or embedded_docs is None:
        print("✗ Skipped (embeddings not available)")
        print()
        return
    
    try:
        embedding_dim = embedding_gen.get_embedding_dimension()
        vector_prep = VectorStorePreparation(embedding_dimension=embedding_dim)
        
        opensearch_data = vector_prep.prepare_for_opensearch(
            documents=embedded_docs,
            index_name="test_documents"
        )
        
        print(f"✓ Prepared {opensearch_data['num_documents']} documents")
        print(f"✓ Index name: {opensearch_data['index_name']}")
        print(f"✓ Embedding dimension: {opensearch_data['embedding_dimension']}")
        
        # Save bulk file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as bulk_file:
            output_path = bulk_file.name
        
        vector_prep.save_opensearch_bulk_file(
            opensearch_data=opensearch_data,
            output_path=output_path
        )
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Document Loader Module - Test Suite")
    print("=" * 60 + "\n")
    
    # Run tests
    documents = test_markdown_loading()
    chunks_recursive = test_recursive_chunking(documents)
    chunks_semantic = test_semantic_chunking(documents)
    embedding_gen, embedded_docs = test_bge_embeddings(chunks_recursive)
    faiss_data, vector_prep = test_faiss_preparation(embedding_gen, embedded_docs)
    test_faiss_search(faiss_data, vector_prep, embedding_gen)
    test_opensearch_preparation(embedding_gen, embedded_docs)
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
