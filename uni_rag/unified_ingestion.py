"""Unified ingestion system supporting multiple sources."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio

from .models import Document
from .docling_ingestion import get_ingestion_client
from .web_ingestion import WebIngestion


class UnifiedIngestion:
    """Unified ingestion supporting files, URLs, and directories."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.file_client = get_ingestion_client(config.get("file", {}))
        self.web_client = WebIngestion(config.get("web", {}))
    
    async def ingest(self, source: Union[str, List[str]], 
                    source_type: str = "auto",
                    metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Unified ingestion method.
        
        Args:
            source: File path, URL, or list of sources
            source_type: "auto", "file", "url", "directory", "sitemap"
            metadata: Additional metadata
        """
        if isinstance(source, list):
            return await self._ingest_multiple(source, source_type, metadata)
        
        # Auto-detect source type
        if source_type == "auto":
            source_type = self._detect_source_type(source)
        
        # Route to appropriate ingestion method
        if source_type == "file":
            return await self.file_client.ingest_file(source, metadata)
        elif source_type == "url":
            return await self.web_client.ingest_url(source, metadata)
        elif source_type == "directory":
            return await self.file_client.ingest_directory(source, metadata=metadata)
        elif source_type == "sitemap":
            return await self.web_client.ingest_sitemap(source, metadata=metadata)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _detect_source_type(self, source: str) -> str:
        """Auto-detect source type from string."""
        if source.startswith(("http://", "https://")):
            if "sitemap" in source.lower():
                return "sitemap"
            return "url"
        
        path = Path(source)
        if path.is_dir():
            return "directory"
        elif path.exists() or path.suffix:
            return "file"
        else:
            # Assume URL if not a valid file path
            return "url"
    
    async def _ingest_multiple(self, sources: List[str], 
                             source_type: str,
                             metadata: Optional[Dict[str, Any]]) -> List[Document]:
        """Ingest multiple sources in parallel."""
        tasks = []
        for source in sources:
            task = self.ingest(source, source_type, metadata)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_documents = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
            elif isinstance(result, Exception):
                print(f"Ingestion error: {result}")
        
        return all_documents
    
    async def ingest_mixed_sources(self, sources: Dict[str, List[str]], 
                                 metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Ingest from mixed source types.
        
        Args:
            sources: {"files": [...], "urls": [...], "directories": [...]}
            metadata: Additional metadata
        """
        all_documents = []
        
        for source_type, source_list in sources.items():
            if source_list:
                docs = await self._ingest_multiple(source_list, source_type.rstrip('s'), metadata)
                all_documents.extend(docs)
        
        return all_documents