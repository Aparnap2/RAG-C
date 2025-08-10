"""
Hybrid retriever with reciprocal rank fusion (RRF).
Combines vector search and BM25 text search for improved retrieval.
"""
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retriever that combines vector search and BM25 text search using RRF.
    """
    def __init__(self, vector_store, text_index, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.text_index = text_index
        self.config = config
        self.rrf_k = config.get("rrf_k", 60)  # RRF constant
        self.vector_weight = config.get("vector_weight", 1.0)
        self.bm25_weight = config.get("bm25_weight", 1.0)
        
    async def retrieve(self, query: str, filters: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search with RRF fusion
        
        Args:
            query: Search query
            filters: Filters to apply (tenant, ACL, time window)
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Run parallel searches
        vector_results_future = self.vector_store.search(query, top_k, filters)
        bm25_results_future = self.text_index.search(query, top_k, filters)
        
        # Gather results
        vector_results, bm25_results = await asyncio.gather(
            vector_results_future, 
            bm25_results_future
        )
        
        # Convert vector results to dict format for RRF
        vector_dict_results = [
            {"id": doc.id, "score": getattr(doc, 'score', 1.0), "doc": doc}
            for doc in vector_results
        ]
        
        # BM25 results are already in dict format
        bm25_dict_results = [
            {"id": result["id"], "score": result["score"], "result": result}
            for result in bm25_results
        ]
        
        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            [
                {"results": vector_dict_results, "weight": self.vector_weight},
                {"results": bm25_dict_results, "weight": self.bm25_weight}
            ],
            k=self.rrf_k
        )
        
        # Deduplicate results
        deduplicated = self._deduplicate_results(fused_results)
        
        # Convert back to document format
        return await self._fetch_documents(deduplicated[:top_k], vector_results, bm25_results)
        
    def _apply_filters(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply filters to query
        
        Args:
            query: Search query
            filters: Filters to apply
            
        Returns:
            Filtered query object
        """
        # Create filtered query object
        filtered_query = {
            "query": query,
            "filters": filters
        }
        
        # Add tenant filter
        if "tenant_id" in filters:
            filtered_query["tenant_id"] = filters["tenant_id"]
            
        # Add ACL filter
        if "acl" in filters:
            filtered_query["acl"] = filters["acl"]
            
        # Add time window filter
        if "time_window" in filters:
            filtered_query["time_window"] = filters["time_window"]
            
        return filtered_query
        
    def _reciprocal_rank_fusion(self, result_lists: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Implement reciprocal rank fusion algorithm
        
        Args:
            result_lists: List of result lists with weights
            k: RRF constant
            
        Returns:
            Fused results
        """
        # Track document scores
        scores = {}
        
        # Process each result list
        for result_list in result_lists:
            results = result_list["results"]
            weight = result_list["weight"]
            
            for rank, result in enumerate(results):
                doc_id = result["id"]
                # RRF formula: weight * 1 / (rank + k)
                score = weight * (1.0 / (rank + k))
                scores[doc_id] = scores.get(doc_id, 0) + score
                
        # Sort by score
        sorted_docs = sorted(
            [{"id": doc_id, "score": score} for doc_id, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_docs
        
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate results by document ID
        
        Args:
            results: List of results
            
        Returns:
            Deduplicated results
        """
        seen = set()
        deduplicated = []
        
        for result in results:
            if result["id"] not in seen:
                seen.add(result["id"])
                deduplicated.append(result)
                
        return deduplicated
        
    async def _fetch_documents(self, results: List[Dict[str, Any]], 
                             vector_results: List = None, 
                             bm25_results: List = None) -> List[Dict[str, Any]]:
        """
        Fetch full documents for results
        
        Args:
            results: List of result IDs and scores
            vector_results: Original vector search results
            bm25_results: Original BM25 search results
            
        Returns:
            List of full documents
        """
        # Create lookup maps
        vector_map = {doc.id: doc for doc in (vector_results or [])}
        bm25_map = {result["id"]: result for result in (bm25_results or [])}
        
        # Build final results
        documents = []
        for result in results:
            doc_id = result["id"]
            score = result["score"]
            
            # Try to get from vector results first, then BM25
            if doc_id in vector_map:
                doc = vector_map[doc_id]
                doc_dict = {
                    "id": doc.id,
                    "text": doc.content,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "tenant_id": doc.tenant_id,
                    "source_tool": doc.source_tool,
                    "source_id": doc.source_id,
                    "ts_source": doc.ts_source,
                    "ts_ingested": doc.ts_ingested,
                    "acl": doc.acl,
                    "score": score
                }
            elif doc_id in bm25_map:
                bm25_doc = bm25_map[doc_id]
                doc_dict = {
                    "id": doc_id,
                    "text": bm25_doc.get("content", ""),
                    "content": bm25_doc.get("content", ""),
                    "metadata": bm25_doc.get("metadata", {}),
                    "tenant_id": bm25_doc.get("tenant_id"),
                    "source_tool": bm25_doc.get("source_tool"),
                    "source_id": bm25_doc.get("source_id"),
                    "ts_source": bm25_doc.get("ts_source"),
                    "ts_ingested": bm25_doc.get("ts_ingested"),
                    "acl": bm25_doc.get("acl", []),
                    "score": score
                }
            else:
                # Fallback - fetch from vector store
                docs = await self.vector_store.get_documents([doc_id])
                if docs:
                    doc = docs[0]
                    doc_dict = {
                        "id": doc.id,
                        "text": doc.content,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "tenant_id": doc.tenant_id,
                        "source_tool": doc.source_tool,
                        "source_id": doc.source_id,
                        "ts_source": doc.ts_source,
                        "ts_ingested": doc.ts_ingested,
                        "acl": doc.acl,
                        "score": score
                    }
                else:
                    continue
                    
            documents.append(doc_dict)
        
        return documents
        
    async def retrieve_with_graph(self, query: str, filters: Dict[str, Any], 
                                graph_client, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve documents with graph-based constraints
        
        Args:
            query: Search query
            filters: Filters to apply
            graph_client: Graph client for entity linking and expansion
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Extract entities from query
        entities = await self._extract_entities(query, graph_client)
        
        if entities:
            # Expand entities in graph (1-2 hop neighborhood)
            expanded_entities = await self._expand_entities(entities, graph_client)
            
            # Add entity constraints to filters
            entity_filters = self._create_entity_filters(expanded_entities)
            combined_filters = {**filters, **entity_filters}
            
            # Retrieve with combined filters
            results = await self.retrieve(query, combined_filters, top_k)
            
            # Add graph context to results
            results = await self._add_graph_context(results, expanded_entities, graph_client)
            
            return results
        else:
            # No entities found, use standard retrieval
            return await self.retrieve(query, filters, top_k)
            
    async def _extract_entities(self, query: str, graph_client) -> List[Dict[str, Any]]:
        """Extract entities from query"""
        # In a real implementation, this would use an entity linker
        # For now, just return an empty list
        return []
        
    async def _expand_entities(self, entities: List[Dict[str, Any]], graph_client) -> List[Dict[str, Any]]:
        """Expand entities in graph (1-2 hop neighborhood)"""
        # In a real implementation, this would query the graph
        # For now, just return the input entities
        return entities
        
    def _create_entity_filters(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create filters from entities"""
        # In a real implementation, this would create filters
        # For now, just return an empty dict
        return {}
        
    async def _add_graph_context(self, results: List[Dict[str, Any]], 
                               entities: List[Dict[str, Any]], 
                               graph_client) -> List[Dict[str, Any]]:
        """Add graph context to results"""
        # In a real implementation, this would add graph context
        # For now, just return the input results
        return results