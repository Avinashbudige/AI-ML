# Document Loader Module - Enterprise Agentic Analytics Assistant (EA³)

A comprehensive document loading and processing module designed for the Enterprise Agentic Analytics Assistant. This module handles unstructured data formats, implements intelligent chunking strategies, generates embeddings, and prepares data for vector database storage.

## Features

### 1. Document Loading
- **PDF Support**: Load PDF documents using PyPDF2
- **Markdown Support**: Load and parse Markdown files
- **Extensible Architecture**: Easy to add support for additional formats

### 2. Chunking Strategies

#### Recursive Chunking
Splits documents into smaller sections based on structural boundaries:
- Paragraph breaks (double newlines)
- Line breaks
- Sentence boundaries (periods, question marks, etc.)
- Word boundaries
- Character-level splitting (as last resort)

#### Semantic Chunking
Identifies meaningful sections based on content structure:
- Header-based splitting (Markdown headers)
- Paragraph grouping
- Topic boundaries
- Sentence-based fallback for large paragraphs

### 3. Embedding Generation

Supports multiple embedding models:
- **OpenAI Embeddings**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- **BGE Models**: BAAI/bge-small-en-v1.5 and other sentence-transformers models

### 4. Vector Store Preparation

Prepare embeddings for storage in:
- **FAISS**: Support for FlatL2, IVFFlat, and HNSW indices
- **OpenSearch**: Generate bulk files and index mappings for OpenSearch

## Installation

### Required Dependencies

```bash
pip install PyPDF2 numpy faiss-cpu sentence-transformers openai
```

For GPU support with FAISS:
```bash
pip install faiss-gpu
```

### Optional Dependencies

```bash
pip install markdown  # For enhanced markdown parsing
```

## Quick Start

### 1. Loading Documents

```python
from document_loader import PDFLoader, MarkdownLoader

# Load a PDF document
pdf_loader = PDFLoader("path/to/document.pdf")
pdf_documents = pdf_loader.load()

# Load a Markdown file
md_loader = MarkdownLoader("path/to/document.md")
md_documents = md_loader.load()

# Load Markdown by sections
md_sections = md_loader.load_by_sections()
```

### 2. Chunking Documents

```python
from document_loader import RecursiveChunker, SemanticChunker

# Recursive chunking
recursive_chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = recursive_chunker.chunk(pdf_documents)

# Semantic chunking
semantic_chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
semantic_chunks = semantic_chunker.chunk(md_documents)
```

### 3. Generating Embeddings

```python
from document_loader import EmbeddingGenerator

# Using OpenAI embeddings (requires API key)
openai_gen = EmbeddingGenerator(
    model_type="openai",
    api_key="your-api-key-here"
)
embedded_docs = openai_gen.embed_documents(chunks)

# Using BGE embeddings (runs locally)
bge_gen = EmbeddingGenerator(
    model_type="bge",
    device="cpu"  # or "cuda" for GPU
)
embedded_docs = bge_gen.embed_documents(chunks)
```

### 4. Preparing for Vector Stores

```python
from document_loader import VectorStorePreparation

# Initialize preparation utility
embedding_dim = bge_gen.get_embedding_dimension()
vector_prep = VectorStorePreparation(embedding_dimension=embedding_dim)

# Prepare for FAISS
faiss_data = vector_prep.prepare_for_faiss(
    documents=embedded_docs,
    index_type="FlatL2"
)

# Save FAISS index
vector_prep.save_faiss_index(
    faiss_data=faiss_data,
    index_path="faiss_index.bin",
    metadata_path="metadata.json"
)

# Prepare for OpenSearch
opensearch_data = vector_prep.prepare_for_opensearch(
    documents=embedded_docs,
    index_name="document_embeddings"
)

# Save OpenSearch bulk file
vector_prep.save_opensearch_bulk_file(
    opensearch_data=opensearch_data,
    output_path="opensearch_bulk.json"
)
```

### 5. Searching with FAISS

```python
# Search for similar documents
query_text = "What is machine learning?"
query_embedding = bge_gen.generate_embeddings([query_text])[0]

similar_docs = vector_prep.search_similar(
    faiss_index=faiss_data['index'],
    query_embedding=query_embedding,
    metadata=faiss_data['metadata'],
    k=5
)

for doc in similar_docs:
    print(f"Score: {doc['score']:.4f}")
    print(f"Content: {doc['document']['content'][:100]}...")
```

## Complete Example

```python
from document_loader import (
    PDFLoader,
    RecursiveChunker,
    EmbeddingGenerator,
    VectorStorePreparation
)

# 1. Load document
loader = PDFLoader("research_paper.pdf")
documents = loader.load()

# 2. Chunk documents
chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(documents)

# 3. Generate embeddings
embedding_gen = EmbeddingGenerator(model_type="bge", device="cpu")
embedded_chunks = embedding_gen.embed_documents(chunks)

# 4. Prepare for vector store
embedding_dim = embedding_gen.get_embedding_dimension()
vector_prep = VectorStorePreparation(embedding_dimension=embedding_dim)

# 5. Create FAISS index
faiss_data = vector_prep.prepare_for_faiss(embedded_chunks, index_type="FlatL2")

# 6. Save index
vector_prep.save_faiss_index(
    faiss_data=faiss_data,
    index_path="my_faiss_index.bin",
    metadata_path="my_metadata.json"
)

print(f"Processed {len(chunks)} chunks from {len(documents)} pages")
print(f"FAISS index created with {faiss_data['num_vectors']} vectors")
```

## Configuration

The module uses configuration settings defined in `config.py`:

```python
# Chunking settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Embedding models
DEFAULT_EMBEDDING_MODEL = "openai"  # or "bge"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
BGE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Vector store settings
DEFAULT_VECTOR_STORE = "faiss"  # or "opensearch"
FAISS_INDEX_TYPE = "FlatL2"  # or "IVFFlat", "HNSW"
```

## Architecture

```
document_loader/
├── __init__.py              # Module initialization
├── config.py                # Configuration settings
├── base_loader.py           # Base document loader class
├── pdf_loader.py            # PDF document loader
├── markdown_loader.py       # Markdown document loader
├── chunking.py              # Chunking strategies
├── embeddings.py            # Embedding generation
├── vector_store.py          # Vector store preparation
└── example_usage.py         # Example usage script
```

## Extending the Module

### Adding a New Document Loader

```python
from document_loader.base_loader import BaseDocumentLoader

class CustomLoader(BaseDocumentLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        # Add validation
        
    def load(self) -> List[Dict[str, Any]]:
        # Implement loading logic
        return documents
```

### Adding a New Chunking Strategy

```python
from document_loader.chunking import BaseChunker

class CustomChunker(BaseChunker):
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implement chunking logic
        return chunks
```

### Adding a New Embedding Model

```python
from document_loader.embeddings import BaseEmbeddingGenerator

class CustomEmbeddingGenerator(BaseEmbeddingGenerator):
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Implement embedding generation
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return dimension
```

## API Reference

### Document Loaders

#### PDFLoader
- `load()`: Load all pages from PDF
- `load_specific_pages(page_numbers)`: Load specific pages

#### MarkdownLoader
- `load()`: Load full markdown document
- `load_by_sections()`: Load document split by headers

### Chunkers

#### RecursiveChunker
- `chunk(documents)`: Split using recursive strategy
- Parameters: `chunk_size`, `chunk_overlap`

#### SemanticChunker
- `chunk(documents)`: Split using semantic strategy
- Parameters: `chunk_size`, `chunk_overlap`

### Embedding Generator

#### EmbeddingGenerator
- `generate_embeddings(texts)`: Generate embeddings for text list
- `embed_documents(documents)`: Embed document chunks
- `get_embedding_dimension()`: Get embedding dimension

### Vector Store Preparation

#### VectorStorePreparation
- `prepare_for_faiss(documents, index_type)`: Prepare FAISS index
- `save_faiss_index(faiss_data, index_path, metadata_path)`: Save FAISS index
- `prepare_for_opensearch(documents, index_name)`: Prepare OpenSearch documents
- `save_opensearch_bulk_file(opensearch_data, output_path)`: Save bulk file
- `search_similar(faiss_index, query_embedding, metadata, k)`: Search similar documents

## Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    loader = PDFLoader("document.pdf")
    documents = loader.load()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid file type: {e}")
except Exception as e:
    print(f"Error loading document: {e}")
```

## Performance Considerations

- **Large PDFs**: Consider loading specific pages or using chunking
- **Batch Embedding**: Generate embeddings in batches for better performance
- **FAISS Index Types**:
  - FlatL2: Exact search, good for small datasets
  - IVFFlat: Faster search, needs training, good for medium datasets
  - HNSW: Fast approximate search, good for large datasets

## Future Enhancements

- Support for additional document formats (DOCX, TXT, HTML)
- Advanced semantic chunking using NLP models
- Support for more embedding models (Cohere, Hugging Face)
- Integration with cloud vector databases (Pinecone, Weaviate)
- Batch processing capabilities
- Document metadata extraction

## License

This module is part of the Enterprise Agentic Analytics Assistant project.

## Support

For issues, questions, or contributions, please refer to the main project repository.
