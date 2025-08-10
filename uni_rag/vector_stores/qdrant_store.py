"""Qdrant vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .base import VectorStoreBase
from ..models import Document

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStoreBase):
    """Qdrant vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = QdrantClient(
            url=config.get("url", "http://localhost:6333"),
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 60)
        )
        self.collection_name = config.get("collection_name", "documents")
        self.vector_size = config.get("vector_size", 768)
        self.distance = Distance.COSINE
        
        # Embedding model will be injected or use external service
        self.embedding_client = config.get("embedding_client")
        self.embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists."""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Qdrant."""
        if not documents:
            return
            
        # Generate embeddings
        texts = [doc.content for doc in documents]
        if self.embedding_client:
            embeddings = await self.embedding_client.embed_documents(texts, self.embedding_model)
        else:
            # Fallback to dummy embeddings for testing
            embeddings = [[0.0] * self.vector_size for _ in texts]
        
        # Create points
        points = []
        for i, doc in enumerate(documents):
            point = PointStruct(
                id=doc.id,
                vector=embeddings[i],
                payload={
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "tenant_id": doc.tenant_id,
                    "source_tool": doc.source_tool,
                    "source_id": doc.source_id,
                    "ts_source": doc.ts_source,
                    "ts_ingested": doc.ts_ingested,
                    "acl": doc.acl
                }
            )
            points.append(point)
        
        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Added {len(documents)} documents to Qdrant")
    
    async def search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents in Qdrant."""
        # Generate query embedding
        if self.embedding_client:
            query_embeddings = await self.embedding_client.embed_documents([query], self.embedding_model)
            query_embedding = query_embeddings[0]
        else:
            # Fallback to dummy embedding for testing
            query_embedding = [0.0] * self.vector_size
        
        # Build filter
        qdrant_filter = None
        if filters:
            conditions = []
            if "tenant_id" in filters:
                conditions.append(FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=filters["tenant_id"])
                ))
            if "acl" in filters:
                for acl in filters["acl"]:
                    conditions.append(FieldCondition(
                        key="acl",
                        match=MatchValue(value=acl)
                    ))
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True
        )
        
        # Convert to documents
        documents = []
        for result in results:
            payload = result.payload
            doc = Document(
                id=str(result.id),
                content=payload["content"],
                metadata=payload.get("metadata", {}),
                tenant_id=payload.get("tenant_id"),
                source_tool=payload.get("source_tool"),
                source_id=payload.get("source_id"),
                ts_source=payload.get("ts_source"),
                ts_ingested=payload.get("ts_ingested"),
                acl=payload.get("acl", [])
            )
            documents.append(doc)
        
        return documents
    
    async def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """Get documents by IDs from Qdrant."""
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=doc_ids,
            with_payload=True
        )
        
        documents = []
        for result in results:
            payload = result.payload
            doc = Document(
                id=str(result.id),
                content=payload["content"],
                metadata=payload.get("metadata", {}),
                tenant_id=payload.get("tenant_id"),
                source_tool=payload.get("source_tool"),
                source_id=payload.get("source_id"),
                ts_source=payload.get("ts_source"),
                ts_ingested=payload.get("ts_ingested"),
                acl=payload.get("acl", [])
            )
            documents.append(doc)
        
        return documents
    
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs from Qdrant."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=doc_ids
        )
        logger.info(f"Deleted {len(doc_ids)} documents from Qdrant")
    
    async def health_check(self) -> bool:
        """Check Qdrant health."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False