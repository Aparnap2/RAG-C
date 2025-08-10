"""
Cross-encoder reranker for improving retrieval precision.
Implements feature extraction, caching, and performance optimizations.
"""
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranker using a cross-encoder model to improve retrieval precision.
    """
    def __init__(self, model_client, cache_client=None, config: Dict[str, Any] = None):
        self.model_client = model_client
        self.cache_client = cache_client
        self.config = config or {}
        self.model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour
        self.batch_size = config.get("batch_size", 16)
        
    async def rerank(self, query: str, candidates: List[Dict[str, Any]], 
                   features: Optional[Dict[str, Any]] = None, 
                   top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder model
        
        Args:
            query: Search query
            candidates: Candidate documents
            features: Additional features for reranking
            top_k: Number of results to return
            
        Returns:
            Reranked candidates
        """
        # Check cache first
        if self.cache_client:
            cache_key = self._compute_cache_key(query, candidates)
            cached_result = await self.cache_client.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for query: {query}")
                return cached_result
                
        # Prepare features
        features = features or {}
        
        # Prepare pairs for scoring
        pairs = []
        for candidate in candidates:
            # Create query-document pair
            pair = {
                "query": query,
                "document": candidate["text"],
                "recency": self._get_recency_feature(candidate, features),
                "entity_overlap": self._get_entity_overlap_feature(candidate, features)
            }
            pairs.append(pair)
            
        # Score pairs in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i+self.batch_size]
            batch_scores = await self._score_batch(batch)
            all_scores.extend(batch_scores)
            
        # Combine with original candidates
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = all_scores[i]
            
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        # Cache result
        if self.cache_client:
            await self.cache_client.set(cache_key, reranked[:top_k], ttl=self.cache_ttl)
            
        return reranked[:top_k]
        
    def _compute_cache_key(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Compute cache key for query and candidates"""
        # Create a deterministic representation of candidates
        candidate_ids = sorted([c["id"] for c in candidates])
        key_data = {
            "query": query,
            "candidate_ids": candidate_ids,
            "model": self.model_name
        }
        
        # Compute hash
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
    def _get_recency_feature(self, candidate: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Get recency feature for a candidate"""
        # Check if recency is provided in features
        if "recency" in features and candidate["id"] in features["recency"]:
            return features["recency"][candidate["id"]]
            
        # Calculate recency from timestamp
        if "ts_source" in candidate:
            try:
                # Parse timestamp
                ts = datetime.fromisoformat(candidate["ts_source"].replace("Z", "+00:00"))
                
                # Calculate age in days
                age_days = (datetime.now() - ts).total_seconds() / (24 * 3600)
                
                # Convert to recency score (1.0 for new, 0.0 for old)
                max_age = 365.0  # 1 year
                recency = max(0.0, 1.0 - (age_days / max_age))
                
                return recency
            except Exception:
                pass
                
        # Default recency
        return 0.5
        
    def _get_entity_overlap_feature(self, candidate: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Get entity overlap feature for a candidate"""
        # Check if entity overlap is provided in features
        if "entity_overlap" in features and candidate["id"] in features["entity_overlap"]:
            return features["entity_overlap"][candidate["id"]]
            
        # Default entity overlap
        return 0.0
        
    async def _score_batch(self, batch: List[Dict[str, Any]]) -> List[float]:
        """Score a batch of query-document pairs"""
        # Extract query-document pairs
        query_doc_pairs = [(pair["query"], pair["document"]) for pair in batch]
        
        # Get base scores from model
        base_scores = await self.model_client.score_pairs(query_doc_pairs, self.model_name)
        
        # Apply feature adjustments
        adjusted_scores = []
        for i, score in enumerate(base_scores):
            # Get features
            recency = batch[i]["recency"]
            entity_overlap = batch[i]["entity_overlap"]
            
            # Apply adjustments (simple linear combination)
            recency_weight = self.config.get("recency_weight", 0.1)
            entity_weight = self.config.get("entity_weight", 0.2)
            
            adjusted_score = score + (recency_weight * recency) + (entity_weight * entity_overlap)
            adjusted_scores.append(adjusted_score)
            
        return adjusted_scores
        
    async def extract_features(self, query: str, candidates: List[Dict[str, Any]], 
                             graph_client=None) -> Dict[str, Any]:
        """
        Extract features for reranking
        
        Args:
            query: Search query
            candidates: Candidate documents
            graph_client: Optional graph client for entity features
            
        Returns:
            Features for reranking
        """
        features = {
            "recency": {},
            "entity_overlap": {}
        }
        
        # Extract recency features
        for candidate in candidates:
            features["recency"][candidate["id"]] = self._get_recency_feature(candidate, {})
            
        # Extract entity features if graph client is provided
        if graph_client:
            # Extract entities from query
            query_entities = await self._extract_entities(query, graph_client)
            
            # Extract entities from candidates
            for candidate in candidates:
                candidate_entities = await self._extract_entities(candidate["text"], graph_client)
                
                # Calculate overlap
                overlap = len(set(query_entities) & set(candidate_entities))
                if query_entities:
                    normalized_overlap = overlap / len(query_entities)
                else:
                    normalized_overlap = 0.0
                    
                features["entity_overlap"][candidate["id"]] = normalized_overlap
                
        return features
        
    async def _extract_entities(self, text: str, graph_client) -> List[str]:
        """Extract entities from text"""
        # In a real implementation, this would use an entity extractor
        # For now, just return an empty list
        return []