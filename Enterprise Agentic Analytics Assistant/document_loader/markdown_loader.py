"""
Markdown document loader.
"""

from typing import List, Dict, Any
import re
from .base_loader import BaseDocumentLoader
from .config import SUPPORTED_MARKDOWN_EXTENSIONS


class MarkdownLoader(BaseDocumentLoader):
    """
    Document loader for Markdown files.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the Markdown loader.
        
        Args:
            file_path (str): Path to the Markdown file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a Markdown file
        """
        super().__init__(file_path)
        
        if not self.validate_file_extension(SUPPORTED_MARKDOWN_EXTENSIONS):
            raise ValueError(
                f"Invalid file extension. Expected one of {SUPPORTED_MARKDOWN_EXTENSIONS}, "
                f"got {self.file_path.suffix}"
            )
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load the Markdown document and extract content.
        
        Returns:
            List[Dict[str, Any]]: List containing the full document with metadata
                Each dict contains:
                - 'content': The text content
                - 'metadata': Dict with file info, headers, etc.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract headers for metadata
            headers = self._extract_headers(content)
            
            document = {
                'content': content,
                'metadata': {
                    'source': str(self.file_path.absolute()),
                    'file_type': 'markdown',
                    'headers': headers,
                    'num_headers': len(headers),
                    **self.get_file_info()
                }
            }
            
            return [document]
            
        except Exception as e:
            raise Exception(f"Error loading Markdown file: {str(e)}")
    
    def load_by_sections(self) -> List[Dict[str, Any]]:
        """
        Load the Markdown document split by top-level headers.
        
        Returns:
            List[Dict[str, Any]]: List of document chunks (one per section)
        """
        documents = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by top-level headers (# Header)
            sections = re.split(r'\n(?=# )', content)
            
            for idx, section in enumerate(sections):
                if section.strip():
                    # Extract the header of this section
                    header_match = re.match(r'#\s+(.+)', section)
                    header = header_match.group(1) if header_match else f"Section {idx + 1}"
                    
                    doc_chunk = {
                        'content': section.strip(),
                        'metadata': {
                            'source': str(self.file_path.absolute()),
                            'section': idx + 1,
                            'header': header,
                            'file_type': 'markdown',
                            **self.get_file_info()
                        }
                    }
                    documents.append(doc_chunk)
                    
        except Exception as e:
            raise Exception(f"Error loading Markdown sections: {str(e)}")
        
        return documents
    
    def _extract_headers(self, content: str) -> List[Dict[str, str]]:
        """
        Extract all headers from the Markdown content.
        
        Args:
            content (str): The Markdown content
            
        Returns:
            List[Dict[str, str]]: List of headers with their levels
        """
        headers = []
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        for match in header_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append({
                'level': level,
                'text': text
            })
        
        return headers
