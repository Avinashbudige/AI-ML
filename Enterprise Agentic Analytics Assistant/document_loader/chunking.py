"""
Chunking strategies for document processing.

This module provides different strategies for splitting documents into smaller,
manageable chunks for processing and embedding generation.
"""

from typing import List, Dict, Any
import re
from abc import ABC, abstractmethod
from .config import CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_SIMILARITY_THRESHOLD


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the chunker.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to chunk
            
        Returns:
            List[Dict[str, Any]]: List of document chunks
        """
        pass


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking strategy that splits documents based on structure.
    
    This chunker attempts to split on:
    1. Paragraph boundaries (double newlines)
    2. Sentence boundaries (periods, question marks, exclamation marks)
    3. Word boundaries (spaces)
    4. Character boundaries (as last resort)
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the recursive chunker.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        
        # Define separators in order of preference
        self.separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",     # Words
            ""       # Characters
        ]
    
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks using recursive splitting.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to chunk
            
        Returns:
            List[Dict[str, Any]]: List of document chunks with metadata
        """
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # Split the content
            chunks = self._recursive_split(content, self.separators)
            
            # Create chunk documents with metadata
            for idx, chunk_text in enumerate(chunks):
                chunk_doc = {
                    'content': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': idx,
                        'chunk_method': 'recursive',
                        'chunk_size': len(chunk_text)
                    }
                }
                all_chunks.append(chunk_doc)
        
        return all_chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the given separators.
        
        Args:
            text (str): Text to split
            separators (List[str]): List of separators to try
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Try each separator in order
        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # Reconstruct with separator
                    if current_chunk:
                        test_chunk = current_chunk + separator + split
                    else:
                        test_chunk = split
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Current chunk is full, save it and start new one
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If the split itself is too large, recursively split it
                        if len(split) > self.chunk_size:
                            remaining_seps = separators[separators.index(separator) + 1:]
                            if remaining_seps:
                                sub_chunks = self._recursive_split(split, remaining_seps)
                                chunks.extend(sub_chunks)
                                current_chunk = ""
                            else:
                                current_chunk = split
                        else:
                            current_chunk = split
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                return self._apply_overlap(chunks)
        
        # If no separator found, return as single chunk or split by characters
        return [text]
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.
        
        Args:
            chunks (List[str]): List of chunks without overlap
            
        Returns:
            List[str]: List of chunks with overlap applied
        """
        if not chunks or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                overlapped_chunks.append(overlap_text + chunk)
        
        return overlapped_chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy that identifies meaningful sections of text.
    
    This chunker splits documents based on semantic boundaries, attempting to
    keep related content together. It uses simple heuristics based on:
    - Headers and subheaders
    - Topic transitions
    - Paragraph structure
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size (int): Target size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
    
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into semantically meaningful chunks.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to chunk
            
        Returns:
            List[Dict[str, Any]]: List of document chunks with metadata
        """
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # Identify semantic boundaries
            chunks = self._identify_semantic_chunks(content)
            
            # Create chunk documents with metadata
            for idx, chunk_text in enumerate(chunks):
                chunk_doc = {
                    'content': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': idx,
                        'chunk_method': 'semantic',
                        'chunk_size': len(chunk_text)
                    }
                }
                all_chunks.append(chunk_doc)
        
        return all_chunks
    
    def _identify_semantic_chunks(self, text: str) -> List[str]:
        """
        Identify semantic boundaries and split text accordingly.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of semantic chunks
        """
        # Split by headers first (Markdown style)
        header_pattern = r'\n(?=#{1,6}\s+)'
        sections = re.split(header_pattern, text)
        
        chunks = []
        for section in sections:
            if not section.strip():
                continue
            
            # If section is within chunk size, keep it as is
            if len(section) <= self.chunk_size:
                chunks.append(section.strip())
            else:
                # Split large sections by paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if not para.strip():
                        continue
                    
                    # Try to add paragraph to current chunk
                    test_chunk = current_chunk + "\n\n" + para if current_chunk else para
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # If paragraph itself is too large, split by sentences
                        if len(para) > self.chunk_size:
                            sentence_chunks = self._split_by_sentences(para)
                            chunks.extend(sentence_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = para
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentences when paragraphs are too large.
        
        Args:
            text (str): Text to split
            
        Returns:
            List[str]: List of sentence-based chunks
        """
        # Simple sentence boundary detection
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
