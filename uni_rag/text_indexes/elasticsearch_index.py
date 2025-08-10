"""Elasticsearch text index implementation for BM25."""

import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch

from .base import TextIndexBase
from ..models import Document

logger = logging.getLogger(__name__)


class ElasticsearchTextIndex(TextIndexBase):
    """Elasticsearch text index for BM25 search."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Build connection parameters
        es_config = {
            'hosts': [config.get('url', 'http://localhost:9200')],
            'timeout': config.get('timeout', 30),
            'max_retries': config.get('max_retries', 3),
            'retry_on_timeout': True
        }
        
        # Add authentication if provided
        if config.get('username') and config.get('password'):
            es_config['basic_auth'] = (config['username'], config['password'])
        elif config.get('api_key'):
            es_config['api_key'] = config['api_key']
        
        # SSL configuration
        if config.get('use_ssl', False):
            es_config['use_ssl'] = True
            es_config['verify_certs'] = config.get('verify_certs', True)
            if config.get('ca_certs'):
                es_config['ca_certs'] = config['ca_certs']
        
        self.client = Elasticsearch(**es_config)
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
            logger.info(f"Created Elasticsearch index: {self.index_name}")
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Elasticsearch."""
        if not documents:
            return
            
        # Prepare bulk operations
        operations = []
        for doc in documents:
            # Index operation
            operations.append({
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "tenant_id": doc.tenant_id,
                    "source_tool": doc.source_tool,
                    "source_id": doc.source_id,
                    "ts_source": doc.ts_source,
                    "ts_ingested": doc.ts_ingested,
                    "acl": doc.acl
                }
            })
        
        # Execute bulk operation
        from elasticsearch.helpers import bulk
        success, failed = bulk(
            self.client,
            operations,
            index=self.index_name,
            raise_on_error=False
        )
        
        if failed:
            logger.error(f"Elasticsearch bulk errors: {failed}")
        
        logger.info(f"Added {success} documents to Elasticsearch")
    
    async def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using BM25 in Elasticsearch."""
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
        """Delete documents by IDs from Elasticsearch."""
        # Prepare bulk delete operations
        operations = []
        for doc_id in doc_ids:
            operations.append({
                "_op_type": "delete",
                "_index": self.index_name,
                "_id": doc_id
            })
        
        # Execute bulk operation
        from elasticsearch.helpers import bulk
        success, failed = bulk(
            self.client,
            operations,
            index=self.index_name,
            raise_on_error=False
        )
        
        if failed:
            logger.error(f"Elasticsearch bulk delete errors: {failed}")
        
        logger.info(f"Deleted {success} documents from Elasticsearch")
    
    async def health_check(self) -> bool:
        """Check Elasticsearch health."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False