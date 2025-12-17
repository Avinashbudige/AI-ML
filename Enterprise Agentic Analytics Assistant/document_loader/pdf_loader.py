"""
PDF document loader using PyPDF2.
"""

from typing import List, Dict, Any
import PyPDF2
from .base_loader import BaseDocumentLoader
from .config import SUPPORTED_PDF_EXTENSIONS


class PDFLoader(BaseDocumentLoader):
    """
    Document loader for PDF files using PyPDF2.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the PDF loader.
        
        Args:
            file_path (str): Path to the PDF file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a PDF
        """
        super().__init__(file_path)
        
        if not self.validate_file_extension(SUPPORTED_PDF_EXTENSIONS):
            raise ValueError(
                f"Invalid file extension. Expected one of {SUPPORTED_PDF_EXTENSIONS}, "
                f"got {self.file_path.suffix}"
            )
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load the PDF document and extract text from all pages.
        
        Returns:
            List[Dict[str, Any]]: List of document chunks (one per page) with metadata
                Each dict contains:
                - 'content': The text content of the page
                - 'metadata': Dict with page number, file info, etc.
        """
        documents = []
        
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Create document chunk with metadata
                    doc_chunk = {
                        'content': text,
                        'metadata': {
                            'source': str(self.file_path.absolute()),
                            'page': page_num + 1,
                            'total_pages': num_pages,
                            'file_type': 'pdf',
                            **self.get_file_info()
                        }
                    }
                    documents.append(doc_chunk)
                    
        except Exception as e:
            raise Exception(f"Error loading PDF file: {str(e)}")
        
        return documents
    
    def load_specific_pages(self, page_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Load specific pages from the PDF document.
        
        Args:
            page_numbers (List[int]): List of page numbers to load (1-indexed)
            
        Returns:
            List[Dict[str, Any]]: List of document chunks for specified pages
        """
        documents = []
        
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in page_numbers:
                    if page_num < 1 or page_num > num_pages:
                        print(f"Warning: Page {page_num} is out of range. Skipping.")
                        continue
                    
                    page = pdf_reader.pages[page_num - 1]  # Convert to 0-indexed
                    text = page.extract_text()
                    
                    doc_chunk = {
                        'content': text,
                        'metadata': {
                            'source': str(self.file_path.absolute()),
                            'page': page_num,
                            'total_pages': num_pages,
                            'file_type': 'pdf',
                            **self.get_file_info()
                        }
                    }
                    documents.append(doc_chunk)
                    
        except Exception as e:
            raise Exception(f"Error loading PDF pages: {str(e)}")
        
        return documents
