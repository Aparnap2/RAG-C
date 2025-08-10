"""OpenSearch text index implementation for BM25."""

import logging
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection

from .base import TextIndexBase
from ..models import Document

logger = logging.getLogger(__name__)


class OpenSearchTextIndex(TextIndexBase):
    """OpenSearch text index for BM25 search."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenSearch(
            hosts=[{
                'host': config.get('host', 'localhost'),
                'port': config.get('port', 9200)
            }],
            http_auth=(config.get('username'), config.get('password')) if config.get('username') else None,
            use_ssl=config.get('use_ssl', False),
            verify_certs=config.get('verify_certs', False),
            connection_class=RequestsHttpConnection,
            timeout=config.get('timeout', 30)
        )
        
        self.index_name = config.get('index_name', 'documents')
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure the index exists with proper mapping."""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "metadata": {"type": "object"},
                        "tenant_id": {"type": "keyword"},
                        "source_tool": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "ts_source": {"type": "date"},
                        "ts_ingested": {"type": "date"},
                        "acl": {"type": "keyword"}
                    }
                }
            }
            
            self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            logger.info(f"Created OpenSearch index: {self.index_name}")
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to OpenSearch."""
        if not documents:
            return
            
        # Prepare bulk operations
        bulk_body = []
        for doc in documents:
            # Index operation
            bulk_body.append({
                "index": {
                    "_index": self.index_name,
                    "_id": doc.id
                }
            })
            
            # Document body
            bulk_body.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "tenant_id": doc.tenant_id,
                "source_tool": doc.source_tool,
                "source_id": doc.source_id,
                "ts_source": doc.ts_source,
                "ts_ingested": doc.ts_ingested,
                "acl": doc.acl
            })
        
        # Execute bulk operation
        response = self.client.bulk(body=bulk_body)
        
        # Check for errors
        if response.get("errors"):
            errors = [item for item in response["items"] if "error" in item.get("index", {})]
            logger.error(f"OpenSearch bulk errors: {errors}")
        
        logger.info(f"Added {len(documents)} documents to OpenSearch")
    
    async def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using BM25 in OpenSearch."""
        # Build query
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "operator": "or"
                                }
                            }
                        }
                    ]
                }
            },
            "size": k,
            "_source": ["content", "metadata", "tenant_id", "source_tool", "source_id", "ts_source", "ts_ingested", "acl"]
        }
        
        # Add filters
        if filters:
            filter_clauses = []
            
            if "tenant_id" in filters:
                filter_clauses.append({
                    "term": {"tenant_id": filters["tenant_id"]}
                })
            
            if "acl" in filters:
                filter_clauses.append({
                    "terms": {"acl": filters["acl"]}
                })
            
            if "time_window" in filters:
                time_filter = filters["time_window"]
                if "start" in time_filter or "end" in time_filter:
                    range_filter = {"range": {"ts_source": {}}}
                    if "start" in time_filter:
                        range_filter["range"]["ts_source"]["gte"] = time_filter["start"]
                    if "end" in time_filter:
                        range_filter["range"]["ts_source"]["lte"] = time_filter["end"]
                    filter_clauses.append(range_filter)
            
            if filter_clauses:
                search_body["query"]["bool"]["filter"] = filter_clauses
        
        # Execute search
        response = self.client.search(
            index=self.index_name,
            body=search_body
        )
        
        # Convert results
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)
        
        return results
    
    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs from OpenSearch."""
        # Prepare bulk delete operations
        bulk_body = []
        for doc_id in doc_ids:
            bulk_body.append({
                "delete": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            })
        
        # Execute bulk operation
        response = self.client.bulk(body=bulk_body)
        
        # Check for errors
        if response.get("errors"):
            errors = [item for item in response["items"] if "error" in item.get("delete", {})]
            logger.error(f"OpenSearch bulk delete errors: {errors}")
        
        logger.info(f"Deleted {len(doc_ids)} documents from OpenSearch")
    
    async def health_check(self) -> bool:
        """Check OpenSearch health."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False