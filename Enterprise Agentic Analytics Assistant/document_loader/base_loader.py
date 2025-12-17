"""
Base document loader class providing the interface for all document loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    
    All document loaders should inherit from this class and implement
    the load() method.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the document loader.
        
        Args:
            file_path (str): Path to the document to load
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load the document and return its content.
        
        Returns:
            List[Dict[str, Any]]: List of document chunks with metadata
                Each dict should contain:
                - 'content': The text content
                - 'metadata': Dict with additional information (page number, etc.)
        """
        pass
    
    def validate_file_extension(self, valid_extensions: List[str]) -> bool:
        """
        Validate that the file has one of the valid extensions.
        
        Args:
            valid_extensions (List[str]): List of valid extensions (e.g., ['.pdf', '.txt'])
            
        Returns:
            bool: True if valid, False otherwise
        """
        return self.file_path.suffix.lower() in valid_extensions
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get basic information about the file.
        
        Returns:
            Dict[str, Any]: Dictionary containing file information
        """
        return {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path.absolute()),
            "file_size": self.file_path.stat().st_size,
            "file_extension": self.file_path.suffix,
        }
