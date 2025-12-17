"""
Vector store preparation module for FAISS and OpenSearch.

This module provides utilities to prepare embedded documents for storage
in vector databases like FAISS and OpenSearch.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)


class VectorStorePreparation:
    """
    Utility class for preparing embedded documents for vector store ingestion.
    """
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize the vector store preparation utility.
        
        Args:
            embedding_dimension (int): Dimension of the embeddings
        """
        self.embedding_dimension = embedding_dimension
    
    def prepare_for_faiss(
        self,
        documents: List[Dict[str, Any]],
        index_type: str = "FlatL2"
    ) -> Dict[str, Any]:
        """
        Prepare embedded documents for FAISS vector store.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with embeddings
            index_type (str): Type of FAISS index ('FlatL2', 'IVFFlat', 'HNSW')
            
        Returns:
            Dict[str, Any]: Dictionary containing FAISS index and metadata
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS package is required. Install it with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        # Extract embeddings and metadata
        embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')
        
        # Create FAISS index based on type
        if index_type == "FlatL2":
            index = faiss.IndexFlatL2(self.embedding_dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)
            # Train the index
            index.train(embeddings)
        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Prepare metadata for each document
        metadata = []
        for idx, doc in enumerate(documents):
            meta = {
                'id': idx,
                'content': doc['content'],
                **doc.get('metadata', {})
            }
            # Remove non-serializable objects
            if 'embedding' in meta:
                del meta['embedding']
            metadata.append(meta)
        
        return {
            'index': index,
            'metadata': metadata,
            'num_vectors': len(embeddings),
            'embedding_dimension': self.embedding_dimension,
            'index_type': index_type
        }
    
    def save_faiss_index(
        self,
        faiss_data: Dict[str, Any],
        index_path: str,
        metadata_path: str
    ) -> Dict[str, str]:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            faiss_data (Dict[str, Any]): FAISS data from prepare_for_faiss()
            index_path (str): Path to save the FAISS index
            metadata_path (str): Path to save the metadata JSON
            
        Returns:
            Dict[str, str]: Paths where files were saved
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS package is required")
        
        # Save FAISS index
        faiss.write_index(faiss_data['index'], index_path)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(faiss_data['metadata'], f, indent=2)
        
        logger.info(f"FAISS index saved to: {index_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return {
            'index_path': index_path,
            'metadata_path': metadata_path
        }
    
    def prepare_for_opensearch(
        self,
        documents: List[Dict[str, Any]],
        index_name: str = "document_embeddings"
    ) -> Dict[str, Any]:
        """
        Prepare embedded documents for OpenSearch vector store.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with embeddings
            index_name (str): Name of the OpenSearch index
            
        Returns:
            Dict[str, Any]: Dictionary containing OpenSearch-formatted documents
        """
        opensearch_docs = []
        
        for idx, doc in enumerate(documents):
            # Convert embedding to list for JSON serialization
            embedding_vector = doc['embedding'].tolist() if isinstance(
                doc['embedding'], np.ndarray
            ) else doc['embedding']
            
            # Prepare document for OpenSearch
            opensearch_doc = {
                'id': idx,
                'content': doc['content'],
                'embedding': embedding_vector,
                'metadata': doc.get('metadata', {})
            }
            opensearch_docs.append(opensearch_doc)
        
        return {
            'index_name': index_name,
            'documents': opensearch_docs,
            'num_documents': len(opensearch_docs),
            'embedding_dimension': self.embedding_dimension
        }
    
    def get_opensearch_index_mapping(self) -> Dict[str, Any]:
        """
        Get the OpenSearch index mapping for vector search.
        
        Returns:
            Dict[str, Any]: OpenSearch index mapping configuration
        """
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib"
                        }
                    },
                    "metadata": {"type": "object", "enabled": True}
                }
            }
        }
    
    def save_opensearch_bulk_file(
        self,
        opensearch_data: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Save OpenSearch documents in bulk format for ingestion.
        
        Args:
            opensearch_data (Dict[str, Any]): OpenSearch data from prepare_for_opensearch()
            output_path (str): Path to save the bulk file
            
        Returns:
            str: Path where bulk file was saved
        """
        index_name = opensearch_data['index_name']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in opensearch_data['documents']:
                # Action line
                action = {"index": {"_index": index_name, "_id": doc['id']}}
                f.write(json.dumps(action) + '\n')
                
                # Document line
                f.write(json.dumps(doc) + '\n')
        
        logger.info(f"OpenSearch bulk file saved to: {output_path}")
        logger.info(f"Use: curl -X POST 'localhost:9200/_bulk' -H 'Content-Type: application/json' --data-binary @{output_path}")
        
        return output_path
    
    def search_similar(
        self,
        faiss_index: Any,
        query_embedding: np.ndarray,
        metadata: List[Dict[str, Any]],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using FAISS index.
        
        Args:
            faiss_index: FAISS index object
            query_embedding (np.ndarray): Query embedding vector
            metadata (List[Dict[str, Any]]): Document metadata
            k (int): Number of similar documents to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        # Ensure query embedding is 2D and float32
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = faiss_index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(metadata):
                result = {
                    'score': float(dist),
                    'document': metadata[idx]
                }
                results.append(result)
        
        return results
