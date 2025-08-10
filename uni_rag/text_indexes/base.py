"""Base text index interface for BM25 search."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models import Document


class TextIndexBase(ABC):
    """Base class for text index implementations."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the text index."""
        pass
    
    @abstractmethod
    async def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using BM25."""
        pass
    
    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the text index is healthy."""
        pass