from abc import ABC, abstractmethod
from rag.models import Document, RAGQuery
from typing import List, Dict, Any

class IngestionBase(ABC):
    @abstractmethod
    async def ingest(self, **kwargs) -> List[Document]: ...

class VectorStoreBase(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Document]): ...
    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[Document]: ...

class KnowledgeGraphBase(ABC):
    @abstractmethod
    async def add_document(self, document: Document): ...
    @abstractmethod
    async def query_relations(self, query: str) -> List[Dict[str, Any]]: ...

class MemoryBase(ABC):
    @abstractmethod
    async def add_context(self, query: RAGQuery, response: str): ...
    @abstractmethod
    async def get_context(self, query: str) -> str: ...

class LLMBase(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> Any: ... 