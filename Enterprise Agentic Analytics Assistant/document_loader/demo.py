#!/usr/bin/env python3
"""
Quick demonstration of the Document Loader Module
"""

import sys
import tempfile
from pathlib import Path

# Add the Enterprise Agentic Analytics Assistant directory to Python path
script_dir = Path(__file__).parent
ea3_dir = script_dir.parent
sys.path.insert(0, str(ea3_dir))

from document_loader import (
    MarkdownLoader,
    RecursiveChunker,
    SemanticChunker,
    config
)

def main():
    print("\n" + "=" * 70)
    print("Document Loader Module - Live Demonstration")
    print("=" * 70 + "\n")
    
    # Create a sample document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        sample_file = f.name
        f.write("""# Enterprise Agentic Analytics Assistant (EA³)

## Overview
The EA³ system provides advanced analytics capabilities using AI agents.

## Features

### Document Processing
- Support for multiple file formats (PDF, Markdown)
- Intelligent text chunking strategies
- Metadata extraction and tracking

### AI Integration
- Embedding generation with multiple models
- Vector database preparation
- Semantic search capabilities

## Getting Started

To use the document loader:
1. Import the required modules
2. Load your documents
3. Apply chunking strategies
4. Generate embeddings
5. Prepare for vector storage

## Technical Details

The system uses state-of-the-art NLP models and vector databases to provide
efficient document retrieval and analysis capabilities.
""")
    
    print(f"📄 Created sample document: {sample_file}\n")
    
    # Step 1: Load the document
    print("Step 1: Loading Markdown Document")
    print("-" * 70)
    loader = MarkdownLoader(sample_file)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} document(s)")
    print(f"✓ Total content length: {len(documents[0]['content'])} characters")
    print(f"✓ Headers found: {len(documents[0]['metadata']['headers'])}")
    for i, header in enumerate(documents[0]['metadata']['headers'][:5]):
        print(f"  {i+1}. Level {header['level']}: {header['text']}")
    print()
    
    # Step 2: Recursive Chunking
    print("Step 2: Applying Recursive Chunking Strategy")
    print("-" * 70)
    recursive_chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    recursive_chunks = recursive_chunker.chunk(documents)
    print(f"✓ Created {len(recursive_chunks)} chunks using recursive strategy")
    print(f"✓ Chunk size: {recursive_chunker.chunk_size} characters")
    print(f"✓ Chunk overlap: {recursive_chunker.chunk_overlap} characters")
    print(f"\nFirst 3 chunks:")
    for i, chunk in enumerate(recursive_chunks[:3]):
        preview = chunk['content'][:80].replace('\n', ' ')
        print(f"  Chunk {i+1} ({chunk['metadata']['chunk_size']} chars): {preview}...")
    print()
    
    # Step 3: Semantic Chunking
    print("Step 3: Applying Semantic Chunking Strategy")
    print("-" * 70)
    semantic_chunker = SemanticChunker(chunk_size=250, chunk_overlap=50)
    semantic_chunks = semantic_chunker.chunk(documents)
    print(f"✓ Created {len(semantic_chunks)} chunks using semantic strategy")
    print(f"✓ Semantic chunking preserves document structure")
    print(f"\nFirst 3 semantic chunks:")
    for i, chunk in enumerate(semantic_chunks[:3]):
        preview = chunk['content'][:80].replace('\n', ' ')
        print(f"  Chunk {i+1} ({chunk['metadata']['chunk_size']} chars): {preview}...")
    print()
    
    # Step 4: Show Configuration
    print("Step 4: Module Configuration")
    print("-" * 70)
    cfg = config.get_config()
    print(f"✓ Default embedding model: {cfg['default_embedding_model']}")
    print(f"✓ Default vector store: {cfg['default_vector_store']}")
    print(f"✓ FAISS index type: {cfg['faiss_index_type']}")
    print(f"✓ Chunk size: {cfg['chunk_size']}")
    print(f"✓ Chunk overlap: {cfg['chunk_overlap']}")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✅ Loaded document with {len(documents[0]['metadata']['headers'])} sections")
    print(f"✅ Recursive chunking: {len(recursive_chunks)} chunks")
    print(f"✅ Semantic chunking: {len(semantic_chunks)} chunks")
    print(f"✅ Ready for embedding generation and vector store preparation")
    print()
    print("Next steps:")
    print("- Generate embeddings using EmbeddingGenerator")
    print("- Prepare for FAISS or OpenSearch with VectorStorePreparation")
    print("- See README.md for complete examples")
    print("=" * 70 + "\n")
    
    # Cleanup
    import os
    os.unlink(sample_file)

if __name__ == "__main__":
    main()
