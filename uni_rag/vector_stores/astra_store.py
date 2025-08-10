"""AstraDB vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
from astrapy import DataAPIClient

from .base import VectorStoreBase
from ..models import Document

logger = logging.getLogger(__name__)


class AstraVectorStore(VectorStoreBase):
    """AstraDB vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize AstraDB client
        client = DataAPIClient(config["application_token"])
        database = client.get_database(
            config["api_endpoint"],
            token=config["application_token"]
        )
        
        self.collection_name = config.get("collection_name", "documents")
        self.vector_dimension = config.get("vector_dimension", 768)
        
        # Get or create collection
        self.collection = database.get_collection(self.collection_name)
        if not self.collection:
            self.collection = database.create_collection(
                self.collection_name,
                dimension=self.vector_dimension,
                metric="cosine"
            )
            logger.info(f"Created AstraDB collection: {self.collection_name}")
        
        # Embedding model will be injected or use external service
        self.embedding_client = config.get("embedding_client")
        self.embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to AstraDB."""
        if not documents:
            return
            
        # Generate embeddings
        texts = [doc.content for doc in documents]
        if self.embedding_client:
            embeddings = await self.embedding_client.embed_documents(texts, self.embedding_model)
        else:
            # Fallback to dummy embeddings for testing
            embeddings = [[0.0] * self.vector_dimension for _ in texts]
        
        # Prepare documents for insertion
        astra_docs = []
        for i, doc in enumerate(documents):
            astra_doc = {
                "_id": doc.id,
                "$vector": embeddings[i],
                "content": doc.content,
                "metadata": doc.metadata,
                "tenant_id": doc.tenant_id,
                "source_tool": doc.source_tool,
                "source_id": doc.source_id,
                "ts_source": doc.ts_source,
                "ts_ingested": doc.ts_ingested,
                "acl": doc.acl
            }
            astra_docs.append(astra_doc)
        
        # Insert documents
        self.collection.insert_many(astra_docs)
        logger.info(f"Added {len(documents)} documents to AstraDB")
    
    async def search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents in AstraDB."""
        # Generate query embedding
        if self.embedding_client:
            query_embeddings = await self.embedding_client.embed_documents([query], self.embedding_model)
            query_embedding = query_embeddings[0]
        else:
            # Fallback to dummy embedding for testing
            query_embedding = [0.0] * self.vector_dimension
        
        # Build filter
        astra_filter = {}
        if filters:
            if "tenant_id" in filters:
                astra_filter["tenant_id"] = filters["tenant_id"]
            if "acl" in filters:
                astra_filter["acl"] = {"$in": filters["acl"]}
        
        # Search
        results = self.collection.vector_find(
            vector=query_embedding,
            limit=k,
            filter=astra_filter if astra_filter else None,
            fields=["content", "metadata", "tenant_id", "source_tool", "source_id", "ts_source", "ts_ingested", "acl"]
        )
        
        # Convert to documents
        documents = []
        for result in results:
            doc = Document(
                id=result["_id"],
                content=result["content"],
                metadata=result.get("metadata", {}),
                tenant_id=result.get("tenant_id"),
                source_tool=result.get("source_tool"),
                source_id=result.get("source_id"),
                ts_source=result.get("ts_source"),
                ts_ingested=result.get("ts_ingested"),
                acl=result.get("acl", [])
            )
            documents.append(doc)
        
        return documents
    
    async def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """Get documents by IDs from AstraDB."""
        results = self.collection.find(
            filter={"_id": {"$in": doc_ids}},
            projection=["content", "metadata", "tenant_id", "source_tool", "source_id", "ts_source", "ts_ingested", "acl"]
        )
        
        documents = []
        for result in results:
            doc = Document(
                id=result["_id"],
                content=result["content"],
                metadata=result.get("metadata", {}),
                tenant_id=result.get("tenant_id"),
                source_tool=result.get("source_tool"),
                source_id=result.get("source_id"),
                ts_source=result.get("ts_source"),
                ts_ingested=result.get("ts_ingested"),
                acl=result.get("acl", [])
            )
            documents.append(doc)
        
        return documents
    
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs from AstraDB."""
        self.collection.delete_many(filter={"_id": {"$in": doc_ids}})
        logger.info(f"Deleted {len(doc_ids)} documents from AstraDB")
    
    async def health_check(self) -> bool:
        """Check AstraDB health."""
        try:
            # Try to find one document to test connection
            list(self.collection.find(limit=1))
            return True
        except Exception as e:
            logger.error(f"AstraDB health check failed: {e}")
            return False