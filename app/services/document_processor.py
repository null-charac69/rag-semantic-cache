"""
Document Processor Service
===========================
Handles extraction and chunking of ALL document types.

Supported formats:
- PDF, DOCX, PPTX, XLSX
- TXT, MD, JSON, CSV
- Code files (.py, .js, .java, etc.)
- HTML

Strategy:
1. Detect file type
2. Extract text using appropriate parser
3. Chunk text intelligently (preserve context)
4. Return structured chunks with metadata
"""

import os
import magic  # python-magic for file type detection
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime
from loguru import logger

# Document parsers
import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation
import pandas as pd
import json
from bs4 import BeautifulSoup
import markdown

from app.models.schemas import DocumentType, DocumentChunk
from app.core.config import settings


class DocumentProcessor:
    """
    Universal document processor.
    
    Handles all document types and returns uniform chunks.
    """
    
    def __init__(self):
        """Initialize document processor."""
        self.supported_extensions = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.pptx': DocumentType.PPTX,
            '.txt': DocumentType.TXT,
            '.md': DocumentType.MD,
            '.csv': DocumentType.CSV,
            '.json': DocumentType.JSON,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.xlsx': DocumentType.XLSX,
            '.xls': DocumentType.XLSX,
            # Code files
            '.py': DocumentType.CODE,
            '.js': DocumentType.CODE,
            '.java': DocumentType.CODE,
            '.cpp': DocumentType.CODE,
            '.c': DocumentType.CODE,
            '.go': DocumentType.CODE,
            '.rs': DocumentType.CODE,
            '.ts': DocumentType.CODE,
            '.jsx': DocumentType.CODE,
            '.tsx': DocumentType.CODE,
        }
        logger.info("✅ Document processor initialized")
    
    
    def detect_file_type(self, file_path: str) -> DocumentType:
        """
        Detect document type from file.
        
        Strategy:
        1. Try extension-based detection (fast)
        2. Fall back to MIME type detection (accurate)
        
        Args:
            file_path: Path to file
            
        Returns:
            DocumentType enum
        """
        # Get file extension
        ext = Path(file_path).suffix.lower()
        
        # Try extension mapping first
        if ext in self.supported_extensions:
            detected_type = self.supported_extensions[ext]
            logger.info(f"Detected type by extension: {detected_type.value}")
            return detected_type
        
        # Fall back to MIME type detection
        try:
            mime = magic.from_file(file_path, mime=True)
            logger.info(f"MIME type detected: {mime}")
            
            mime_mapping = {
                'application/pdf': DocumentType.PDF,
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': DocumentType.PPTX,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentType.XLSX,
                'text/plain': DocumentType.TXT,
                'text/markdown': DocumentType.MD,
                'text/html': DocumentType.HTML,
                'application/json': DocumentType.JSON,
                'text/csv': DocumentType.CSV,
            }
            
            return mime_mapping.get(mime, DocumentType.UNKNOWN)
            
        except Exception as e:
            logger.warning(f"Could not detect MIME type: {e}")
            return DocumentType.UNKNOWN
    
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Uses PyPDF2 - handles most PDFs.
        For OCR PDFs (scanned images), would need pytesseract.
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"Processing PDF with {num_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Add page metadata
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise ValueError(f"Failed to process PDF: {e}")
    
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file.
        
        Handles:
        - Paragraphs
        - Tables
        - Preserves some structure
        """
        try:
            doc = DocxDocument(file_path)
            text_parts = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text for cell in row.cells)
                    text_parts.append(row_text)
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting DOCX: {e}")
            raise ValueError(f"Failed to process DOCX: {e}")
    
    
    def extract_text_from_pptx(self, file_path: str) -> str:
        """
        Extract text from PowerPoint file.
        
        Extracts:
        - Slide titles
        - Slide content
        - Notes (if any)
        """
        try:
            prs = Presentation(file_path)
            text_parts = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = f"\n--- Slide {slide_num} ---\n"
                
                # Extract all text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text += shape.text + "\n"
                
                text_parts.append(slide_text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting PPTX: {e}")
            raise ValueError(f"Failed to process PPTX: {e}")
    
    
    def extract_text_from_xlsx(self, file_path: str) -> str:
        """
        Extract text from Excel file.
        
        Strategy:
        - Read all sheets
        - Convert to readable format
        - Preserve column headers
        """
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert to string representation
                sheet_text = f"\n--- Sheet: {sheet_name} ---\n"
                sheet_text += df.to_string(index=False)
                text_parts.append(sheet_text)
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting XLSX: {e}")
            raise ValueError(f"Failed to process XLSX: {e}")
    
    
    def extract_text_from_csv(self, file_path: str) -> str:
        """
        Extract text from CSV file.
        
        Returns readable table format.
        """
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
            
        except Exception as e:
            logger.error(f"Error extracting CSV: {e}")
            raise ValueError(f"Failed to process CSV: {e}")
    
    
    def extract_text_from_json(self, file_path: str) -> str:
        """
        Extract text from JSON file.
        
        Strategy:
        - Pretty print JSON
        - Make it readable for LLM
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to pretty formatted string
            return json.dumps(data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            raise ValueError(f"Failed to process JSON: {e}")
    
    
    def extract_text_from_html(self, file_path: str) -> str:
        """
        Extract text from HTML file.
        
        Uses BeautifulSoup to:
        - Remove scripts/styles
        - Extract clean text
        - Preserve some structure
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting HTML: {e}")
            raise ValueError(f"Failed to process HTML: {e}")
    
    
    def extract_text_from_markdown(self, file_path: str) -> str:
        """
        Extract text from Markdown file.
        
        Strategy:
        - Read as plain text (markdown is readable as-is)
        - Optionally convert to HTML then extract (preserves structure)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
            
        except Exception as e:
            logger.error(f"Error extracting Markdown: {e}")
            raise ValueError(f"Failed to process Markdown: {e}")
    
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
            
        except Exception as e:
            logger.error(f"Error extracting TXT: {e}")
            raise ValueError(f"Failed to process TXT: {e}")
    
    
    def extract_text_from_code(self, file_path: str) -> str:
        """
        Extract text from code file.
        
        Strategy:
        - Read as plain text
        - Add file extension as context
        """
        try:
            ext = Path(file_path).suffix
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Add context about the language
            return f"```{ext[1:]}\n{code}\n```"
            
        except Exception as e:
            logger.error(f"Error extracting code: {e}")
            raise ValueError(f"Failed to process code file: {e}")
    
    
    def extract_text(self, file_path: str, doc_type: DocumentType) -> str:
        """
        Main extraction dispatcher.
        
        Routes to appropriate extractor based on type.
        
        Args:
            file_path: Path to document
            doc_type: Document type
            
        Returns:
            Extracted text
        """
        extractors = {
            DocumentType.PDF: self.extract_text_from_pdf,
            DocumentType.DOCX: self.extract_text_from_docx,
            DocumentType.PPTX: self.extract_text_from_pptx,
            DocumentType.XLSX: self.extract_text_from_xlsx,
            DocumentType.CSV: self.extract_text_from_csv,
            DocumentType.JSON: self.extract_text_from_json,
            DocumentType.HTML: self.extract_text_from_html,
            DocumentType.MD: self.extract_text_from_markdown,
            DocumentType.TXT: self.extract_text_from_txt,
            DocumentType.CODE: self.extract_text_from_code,
        }
        
        extractor = extractors.get(doc_type)
        
        if not extractor:
            raise ValueError(f"No extractor available for type: {doc_type}")
        
        logger.info(f"Extracting text from {doc_type.value} file...")
        text = extractor(file_path)
        logger.success(f"✅ Extracted {len(text)} characters")
        
        return text
    
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Why overlap?
        - Prevents losing context at boundaries
        - Example: "...machine learning. It is..." 
          → Both chunks contain "machine learning" context
        
        Args:
            text: Full document text
            chunk_size: Characters per chunk (default from config)
            chunk_overlap: Overlap between chunks (default from config)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk from start to start+chunk_size
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary (better context preservation)
            if end < len(text):
                # Look for last period, newline, or space
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_space = chunk.rfind(' ')
                
                # Use the latest boundary found
                boundary = max(last_period, last_newline, last_space)
                
                if boundary > chunk_size * 0.5:  # Only if boundary is not too early
                    chunk = chunk[:boundary + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            
            # Move start forward, accounting for overlap
            start = end - chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    
    def process_document(
        self,
        file_path: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Complete document processing pipeline.
        
        Steps:
        1. Detect file type
        2. Extract text
        3. Chunk text
        4. Create DocumentChunk objects with metadata
        
        Args:
            file_path: Path to file
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        logger.info(f"Processing document: {filename}")
        
        # Step 1: Detect type
        doc_type = self.detect_file_type(file_path)
        logger.info(f"Document type: {doc_type.value}")
        
        # Step 2: Extract text
        text = self.extract_text(file_path, doc_type)
        
        # Step 3: Chunk text
        chunks = self.chunk_text(text)
        
        # Step 4: Create DocumentChunk objects
        document_id = self._generate_document_id(filename)
        
        document_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx}"
            
            chunk_metadata = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "document_type": doc_type.value,
                "created_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            )
            
            document_chunks.append(chunk)
        
        logger.success(
            f"✅ Document processed: {len(chunks)} chunks, "
            f"{len(text)} total characters"
        )
        
        return document_chunks
    
    
    def _generate_document_id(self, filename: str) -> str:
        """
        Generate unique document ID.
        
        Uses hash of filename + timestamp for uniqueness.
        """
        unique_string = f"{filename}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_document_processor_instance = None

def get_document_processor() -> DocumentProcessor:
    """
    Get or create singleton document processor.
    
    Usage:
        from app.services.document_processor import get_document_processor
        
        processor = get_document_processor()
        chunks = processor.process_document(file_path, filename)
    """
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance
