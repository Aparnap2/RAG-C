"""Base vector store interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models import Document


class VectorStoreBase(ABC):
    """Base class for vector store implementations."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        pass
    
    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass