"""
Example usage script for the Document Loader module.

This script demonstrates how to use the various components of the document loader
to process documents, generate embeddings, and prepare data for vector stores.
"""

import os
import sys
from pathlib import Path

# Add the Enterprise Agentic Analytics Assistant directory to Python path
script_dir = Path(__file__).parent
ea3_dir = script_dir.parent
root_dir = ea3_dir.parent
sys.path.insert(0, str(root_dir))


def example_pdf_loading():
    """Example: Loading PDF documents."""
    print("=" * 60)
    print("Example 1: Loading PDF Documents")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import PDFLoader
    
    # Note: Replace with actual PDF file path
    pdf_path = "sample_document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Note: PDF file '{pdf_path}' not found. Skipping this example.")
        print("To run this example, place a PDF file at the specified path.\n")
        return None
    
    try:
        # Load PDF
        loader = PDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages from PDF")
        print(f"First page preview: {documents[0]['content'][:200]}...")
        print(f"Metadata: {documents[0]['metadata']}\n")
        
        return documents
        
    except Exception as e:
        print(f"Error: {e}\n")
        return None


def example_markdown_loading():
    """Example: Loading Markdown documents."""
    print("=" * 60)
    print("Example 2: Loading Markdown Documents")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import MarkdownLoader
    
    # Create a sample markdown file for demonstration
    sample_md_path = "/tmp/sample_document.md"
    sample_content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on learning from data.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models.

### Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data.

## Applications

Machine learning has many applications including:
- Image recognition
- Natural language processing
- Recommendation systems
"""
    
    # Write sample file
    with open(sample_md_path, 'w') as f:
        f.write(sample_content)
    
    try:
        # Load Markdown
        loader = MarkdownLoader(sample_md_path)
        documents = loader.load()
        
        print(f"Loaded markdown document")
        print(f"Content preview: {documents[0]['content'][:200]}...")
        print(f"Headers found: {documents[0]['metadata']['headers'][:3]}\n")
        
        # Load by sections
        sections = loader.load_by_sections()
        print(f"Document split into {len(sections)} sections")
        for i, section in enumerate(sections):
            print(f"Section {i+1}: {section['metadata'].get('header', 'N/A')}")
        print()
        
        return documents
        
    except Exception as e:
        print(f"Error: {e}\n")
        return None


def example_recursive_chunking():
    """Example: Recursive chunking strategy."""
    print("=" * 60)
    print("Example 3: Recursive Chunking")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import MarkdownLoader, RecursiveChunker
    
    # Load sample markdown
    sample_md_path = "/tmp/sample_document.md"
    if not Path(sample_md_path).exists():
        print("Sample markdown not found. Run example 2 first.\n")
        return None
    
    try:
        loader = MarkdownLoader(sample_md_path)
        documents = loader.load()
        
        # Chunk with recursive strategy
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(documents)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
        print(f"\nFirst chunk:")
        print(f"Content: {chunks[0]['content'][:150]}...")
        print(f"Metadata: chunk_index={chunks[0]['metadata']['chunk_index']}, "
              f"size={chunks[0]['metadata']['chunk_size']}\n")
        
        return chunks
        
    except Exception as e:
        print(f"Error: {e}\n")
        return None


def example_semantic_chunking():
    """Example: Semantic chunking strategy."""
    print("=" * 60)
    print("Example 4: Semantic Chunking")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import MarkdownLoader, SemanticChunker
    
    sample_md_path = "/tmp/sample_document.md"
    if not Path(sample_md_path).exists():
        print("Sample markdown not found. Run example 2 first.\n")
        return None
    
    try:
        loader = MarkdownLoader(sample_md_path)
        documents = loader.load()
        
        # Chunk with semantic strategy
        chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk(documents)
        
        print(f"Created {len(chunks)} semantic chunks from {len(documents)} document(s)")
        print(f"\nFirst chunk:")
        print(f"Content: {chunks[0]['content'][:150]}...")
        print(f"Metadata: chunk_index={chunks[0]['metadata']['chunk_index']}\n")
        
        return chunks
        
    except Exception as e:
        print(f"Error: {e}\n")
        return None


def example_bge_embeddings():
    """Example: Generate embeddings using BGE model."""
    print("=" * 60)
    print("Example 5: BGE Embeddings (Local)")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import EmbeddingGenerator
    
    # Sample texts
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text."
    ]
    
    try:
        # Initialize BGE embedding generator
        embedding_gen = EmbeddingGenerator(
            model_type="bge",
            device="cpu"
        )
        
        # Generate embeddings
        embeddings = embedding_gen.generate_embeddings(texts)
        
        print(f"Generated embeddings for {len(texts)} texts")
        print(f"Embedding dimension: {embedding_gen.get_embedding_dimension()}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"First embedding (first 5 values): {embeddings[0][:5]}\n")
        
        return embedding_gen, embeddings
        
    except ImportError as e:
        print(f"Note: {e}")
        print("Install with: pip install sentence-transformers\n")
        return None, None
    except Exception as e:
        print(f"Error: {e}\n")
        return None, None


def example_openai_embeddings():
    """Example: Generate embeddings using OpenAI."""
    print("=" * 60)
    print("Example 6: OpenAI Embeddings (API-based)")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import EmbeddingGenerator
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Note: OPENAI_API_KEY not set. Skipping this example.")
        print("Set the environment variable to run: export OPENAI_API_KEY='your-key'\n")
        return None, None
    
    texts = [
        "Artificial intelligence is transforming industries.",
        "Machine learning models learn from data patterns."
    ]
    
    try:
        # Initialize OpenAI embedding generator
        embedding_gen = EmbeddingGenerator(
            model_type="openai",
            api_key=api_key
        )
        
        # Generate embeddings
        embeddings = embedding_gen.generate_embeddings(texts)
        
        print(f"Generated embeddings for {len(texts)} texts")
        print(f"Embedding dimension: {embedding_gen.get_embedding_dimension()}")
        print(f"Embeddings shape: {embeddings.shape}\n")
        
        return embedding_gen, embeddings
        
    except ImportError as e:
        print(f"Note: {e}")
        print("Install with: pip install openai\n")
        return None, None
    except Exception as e:
        print(f"Error: {e}\n")
        return None, None


def example_faiss_preparation():
    """Example: Prepare embeddings for FAISS."""
    print("=" * 60)
    print("Example 7: FAISS Vector Store Preparation")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import VectorStorePreparation, EmbeddingGenerator
    
    try:
        # Generate some sample embedded documents
        embedding_gen = EmbeddingGenerator(model_type="bge", device="cpu")
        
        # Sample documents
        sample_docs = [
            {'content': 'Machine learning uses algorithms to learn from data.'},
            {'content': 'Deep learning is a subset of machine learning.'},
            {'content': 'Neural networks are inspired by the human brain.'}
        ]
        
        # Embed documents
        embedded_docs = embedding_gen.embed_documents(sample_docs)
        
        # Prepare for FAISS
        embedding_dim = embedding_gen.get_embedding_dimension()
        vector_prep = VectorStorePreparation(embedding_dimension=embedding_dim)
        
        faiss_data = vector_prep.prepare_for_faiss(
            documents=embedded_docs,
            index_type="FlatL2"
        )
        
        print(f"Created FAISS index with {faiss_data['num_vectors']} vectors")
        print(f"Embedding dimension: {faiss_data['embedding_dimension']}")
        print(f"Index type: {faiss_data['index_type']}")
        
        # Save to temporary location
        vector_prep.save_faiss_index(
            faiss_data=faiss_data,
            index_path="/tmp/sample_faiss_index.bin",
            metadata_path="/tmp/sample_metadata.json"
        )
        print()
        
        return faiss_data, vector_prep, embedding_gen
        
    except ImportError as e:
        print(f"Note: {e}")
        print("Install with: pip install faiss-cpu sentence-transformers\n")
        return None, None, None
    except Exception as e:
        print(f"Error: {e}\n")
        return None, None, None


def example_faiss_search():
    """Example: Search using FAISS index."""
    print("=" * 60)
    print("Example 8: FAISS Similarity Search")
    print("=" * 60)
    
    result = example_faiss_preparation()
    if result[0] is None:
        return
    
    faiss_data, vector_prep, embedding_gen = result
    
    try:
        # Create a query
        query = "What is deep learning?"
        query_embedding = embedding_gen.generate_embeddings([query])[0]
        
        # Search for similar documents
        results = vector_prep.search_similar(
            faiss_index=faiss_data['index'],
            query_embedding=query_embedding,
            metadata=faiss_data['metadata'],
            k=2
        )
        
        print(f"Query: {query}")
        print(f"\nTop {len(results)} similar documents:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.4f}")
            print(f"   Content: {result['document']['content']}")
        print()
        
    except Exception as e:
        print(f"Error: {e}\n")


def example_opensearch_preparation():
    """Example: Prepare embeddings for OpenSearch."""
    print("=" * 60)
    print("Example 9: OpenSearch Preparation")
    print("=" * 60)
    
    from 'Enterprise Agentic Analytics Assistant'.document_loader import VectorStorePreparation, EmbeddingGenerator
    
    try:
        # Generate some sample embedded documents
        embedding_gen = EmbeddingGenerator(model_type="bge", device="cpu")
        
        sample_docs = [
            {'content': 'AI is revolutionizing healthcare.'},
            {'content': 'Machine learning improves predictions.'},
        ]
        
        embedded_docs = embedding_gen.embed_documents(sample_docs)
        
        # Prepare for OpenSearch
        embedding_dim = embedding_gen.get_embedding_dimension()
        vector_prep = VectorStorePreparation(embedding_dimension=embedding_dim)
        
        opensearch_data = vector_prep.prepare_for_opensearch(
            documents=embedded_docs,
            index_name="sample_documents"
        )
        
        print(f"Prepared {opensearch_data['num_documents']} documents for OpenSearch")
        print(f"Index name: {opensearch_data['index_name']}")
        print(f"Embedding dimension: {opensearch_data['embedding_dimension']}")
        
        # Get index mapping
        mapping = vector_prep.get_opensearch_index_mapping()
        print(f"\nIndex mapping: {mapping['mappings']['properties'].keys()}")
        
        # Save bulk file
        vector_prep.save_opensearch_bulk_file(
            opensearch_data=opensearch_data,
            output_path="/tmp/opensearch_bulk.json"
        )
        print()
        
    except ImportError as e:
        print(f"Note: {e}")
        print("Install with: pip install sentence-transformers\n")
    except Exception as e:
        print(f"Error: {e}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Document Loader Module - Example Usage")
    print("=" * 60 + "\n")
    
    # Run examples
    example_markdown_loading()
    example_recursive_chunking()
    example_semantic_chunking()
    example_bge_embeddings()
    example_openai_embeddings()
    example_faiss_preparation()
    example_faiss_search()
    example_opensearch_preparation()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
