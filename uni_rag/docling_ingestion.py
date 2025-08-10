"""Document ingestion using Docling for advanced document processing."""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from .models import Document


class DoclingIngestion:
    """Document ingestion using Docling for PDF, Word, PowerPoint, etc."""
    
    def __init__(self, config: Dict[str, Any]):
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling not available. Install with: pip install docling")
            
        self.config = config
        self.converter = DocumentConverter()
        
    async def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest a single file using Docling."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Convert document
        try:
            result = self.converter.convert(file_path)
            doc_content = result.document
            
            # Extract text content
            text_content = doc_content.export_to_markdown()
            
            # Create document
            document = Document(
                id=f"docling_{path.stem}_{hash(text_content) % 10000}",
                content=text_content,
                metadata={
                    "filename": path.name,
                    "file_type": path.suffix.lower(),
                    "file_size": path.stat().st_size,
                    "docling_metadata": {
                        "page_count": getattr(doc_content, 'page_count', 0),
                        "title": getattr(doc_content, 'title', ''),
                        "author": getattr(doc_content, 'author', ''),
                    },
                    **(metadata or {})
                },
                source_tool="docling",
                source_id=str(path),
                ts_source=str(path.stat().st_mtime),
                ts_ingested=str(asyncio.get_event_loop().time())
            )
            
            return [document]
            
        except Exception as e:
            raise RuntimeError(f"Failed to process {file_path}: {str(e)}")
    
    async def ingest_directory(self, directory_path: str, 
                             file_patterns: List[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest all supported files in a directory."""
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Default supported file patterns
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.txt", "*.md"]
        
        # Find all matching files
        files = []
        for pattern in file_patterns:
            files.extend(path.glob(pattern))
        
        # Process files in parallel (with limit)
        semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 5))
        
        async def process_file(file_path):
            async with semaphore:
                try:
                    return await self.ingest_file(str(file_path), metadata)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return []
        
        # Process all files
        tasks = [process_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_documents = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
        
        return all_documents
    
    async def ingest_url(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest document from URL (if supported by Docling)."""
        try:
            result = self.converter.convert(url)
            doc_content = result.document
            
            # Extract text content
            text_content = doc_content.export_to_markdown()
            
            # Create document
            document = Document(
                id=f"docling_url_{hash(url) % 10000}",
                content=text_content,
                metadata={
                    "url": url,
                    "docling_metadata": {
                        "title": getattr(doc_content, 'title', ''),
                        "author": getattr(doc_content, 'author', ''),
                    },
                    **(metadata or {})
                },
                source_tool="docling",
                source_id=url,
                ts_ingested=str(asyncio.get_event_loop().time())
            )
            
            return [document]
            
        except Exception as e:
            raise RuntimeError(f"Failed to process URL {url}: {str(e)}")


# Fallback simple ingestion if Docling not available
class SimpleFileIngestion:
    """Simple file ingestion fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Simple text file ingestion."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read text content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        document = Document(
            id=f"simple_{path.stem}_{hash(content) % 10000}",
            content=content,
            metadata={
                "filename": path.name,
                "file_type": path.suffix.lower(),
                "file_size": path.stat().st_size,
                **(metadata or {})
            },
            source_tool="simple_file",
            source_id=str(path),
            ts_source=str(path.stat().st_mtime),
            ts_ingested=str(asyncio.get_event_loop().time())
        )
        
        return [document]


def get_ingestion_client(config: Dict[str, Any]):
    """Get appropriate ingestion client."""
    if DOCLING_AVAILABLE and config.get("use_docling", True):
        return DoclingIngestion(config)
    else:
        return SimpleFileIngestion(config)