"""
Embedding generation module for converting text chunks into vector embeddings.

Supports multiple embedding models including OpenAI embeddings and BGE models.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    BGE_EMBEDDING_MODEL,
    OPENAI_API_KEY
)


class BaseEmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.
    """
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings (shape: [num_texts, embedding_dim])
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this generator.
        
        Returns:
            int: Embedding dimension
        """
        pass


class OpenAIEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Embedding generator using OpenAI's embedding models.
    """
    
    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL, api_key: Optional[str] = None):
        """
        Initialize the OpenAI embedding generator.
        
        Args:
            model_name (str): Name of the OpenAI embedding model to use
            api_key (Optional[str]): OpenAI API key. If None, uses OPENAI_API_KEY from config
        """
        self.model_name = model_name
        self.api_key = api_key or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            )
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI's API.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            raise Exception(f"Error generating OpenAI embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of OpenAI embeddings.
        
        Returns:
            int: Embedding dimension (1536 for text-embedding-ada-002)
        """
        # text-embedding-ada-002 produces 1536-dimensional embeddings
        if "ada-002" in self.model_name:
            return 1536
        # text-embedding-3-small produces 1536-dimensional embeddings
        elif "3-small" in self.model_name:
            return 1536
        # text-embedding-3-large produces 3072-dimensional embeddings
        elif "3-large" in self.model_name:
            return 3072
        else:
            return 1536  # Default


class BGEEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Embedding generator using BGE (BAAI General Embedding) models.
    """
    
    def __init__(self, model_name: str = BGE_EMBEDDING_MODEL, device: str = "cpu"):
        """
        Initialize the BGE embedding generator.
        
        Args:
            model_name (str): Name of the BGE model to use
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for BGE embeddings. "
                "Install it with: pip install sentence-transformers"
            )
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using BGE model.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error generating BGE embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of BGE embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


class EmbeddingGenerator:
    """
    Main embedding generator class that provides a unified interface for different models.
    """
    
    def __init__(
        self,
        model_type: str = DEFAULT_EMBEDDING_MODEL,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_type (str): Type of embedding model ('openai' or 'bge')
            model_name (Optional[str]): Specific model name to use
            **kwargs: Additional arguments passed to the specific generator
        """
        self.model_type = model_type.lower()
        
        if self.model_type == "openai":
            model_name = model_name or OPENAI_EMBEDDING_MODEL
            self.generator = OpenAIEmbeddingGenerator(model_name, **kwargs)
        elif self.model_type == "bge":
            model_name = model_name or BGE_EMBEDDING_MODEL
            self.generator = BGEEmbeddingGenerator(model_name, **kwargs)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: 'openai', 'bge'"
            )
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        return self.generator.generate_embeddings(texts)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            documents (List[Dict[str, Any]]): List of document chunks
            
        Returns:
            List[Dict[str, Any]]: Documents with embeddings added to metadata
        """
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to documents
        embedded_documents = []
        for doc, embedding in zip(documents, embeddings):
            embedded_doc = {
                'content': doc['content'],
                'embedding': embedding,
                'metadata': doc.get('metadata', {})
            }
            embedded_documents.append(embedded_doc)
        
        return embedded_documents
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this generator.
        
        Returns:
            int: Embedding dimension
        """
        return self.generator.get_embedding_dimension()
