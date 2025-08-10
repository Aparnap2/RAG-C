"""
Grounded generation with citations.
Implements evidence-based claim verification and citation tracking.
"""
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class GroundedGenerator:
    """
    Generator that produces grounded responses with citations.
    """
    def __init__(self, llm_client, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.min_evidence_score = config.get("min_evidence_score", 0.7)
        self.citation_pattern = re.compile(r'\[(\d+)\]')
        
    async def generate_with_citations(self, query: str, context: List[Dict[str, Any]], 
                                    tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate grounded response with citations
        
        Args:
            query: User query
            context: Retrieved context (chunks and graph facts)
            tenant_id: Optional tenant identifier
            
        Returns:
            Response with answer and citations
        """
        # Check if context is sufficient
        evidence_score = self._calculate_evidence_score(query, context)
        if evidence_score < self.min_evidence_score:
            return {
                "answer": "I don't have enough information to answer that question.",
                "citations": [],
                "has_sufficient_evidence": False,
                "evidence_score": evidence_score
            }
            
        # Prepare context with citation markers
        marked_context = self._mark_context_for_citations(context)
        
        # Prepare prompt
        prompt = f"""
        Answer the query based ONLY on the provided context.
        For each claim in your answer, cite the specific source using [number].
        If the context doesn't contain enough information, say so.
        
        Context:
        {marked_context}
        
        Query: {query}
        """
        
        # Generate response
        response = await self.llm_client.generate(prompt)
        
        # Extract and validate citations
        answer, citations = self._extract_citations(response, context)
        
        return {
            "answer": answer,
            "citations": citations,
            "has_sufficient_evidence": True,
            "evidence_score": evidence_score
        }
        
    async def generate_stream(self, query: str, context: List[Dict[str, Any]], 
                           tenant_id: Optional[str] = None):
        """
        Generate streaming response with citations
        
        Args:
            query: User query
            context: Retrieved context
            tenant_id: Optional tenant identifier
            
        Yields:
            Streaming response chunks
        """
        # Check if context is sufficient
        evidence_score = self._calculate_evidence_score(query, context)
        if evidence_score < self.min_evidence_score:
            yield {
                "type": "answer",
                "content": "I don't have enough information to answer that question.",
                "done": True
            }
            return
            
        # Prepare context with citation markers
        marked_context = self._mark_context_for_citations(context)
        
        # Prepare prompt
        prompt = f"""
        Answer the query based ONLY on the provided context.
        For each claim in your answer, cite the specific source using [number].
        If the context doesn't contain enough information, say so.
        
        Context:
        {marked_context}
        
        Query: {query}
        """
        
        # Generate streaming response
        async for chunk in self.llm_client.generate_stream(prompt):
            yield {
                "type": "answer",
                "content": chunk,
                "done": False
            }
            
        # Send citations at the end
        citations = self._extract_citations_from_context(context)
        yield {
            "type": "citations",
            "content": citations,
            "done": True
        }
        
    def _calculate_evidence_score(self, query: str, context: List[Dict[str, Any]]) -> float:
        """
        Calculate evidence score for query and context
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Evidence score (0.0 to 1.0)
        """
        # In a real implementation, this would use a more sophisticated method
        # For now, just use a simple heuristic based on context length
        
        if not context:
            return 0.0
            
        # Calculate total content length
        total_length = sum(len(item.get("text", "")) for item in context)
        
        # Normalize to a score between 0 and 1
        max_length = 10000  # Maximum expected context length
        score = min(1.0, total_length / max_length)
        
        return score
        
    def _mark_context_for_citations(self, context: List[Dict[str, Any]]) -> str:
        """
        Mark context with citation numbers
        
        Args:
            context: Retrieved context
            
        Returns:
            Marked context string
        """
        marked_context = []
        
        for i, item in enumerate(context):
            if "type" in item and item["type"] == "edge":
                # Format graph edge
                relation = item.get("relation", "")
                t_valid_start = item.get("t_valid_start", "")
                t_valid_end = item.get("t_valid_end", "")
                marked_context.append(f"[{i+1}] {relation} (valid from {t_valid_start} to {t_valid_end})")
            else:
                # Format text chunk
                text = item.get("text", "")
                marked_context.append(f"[{i+1}] {text}")
                
        return "\n\n".join(marked_context)
        
    def _extract_citations(self, response: str, context: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract citations from response
        
        Args:
            response: Generated response
            context: Retrieved context
            
        Returns:
            Tuple of (cleaned answer, citations)
        """
        # Find all citation markers
        citation_markers = set(int(m) for m in self.citation_pattern.findall(response))
        
        # Create citation objects
        citations = []
        for marker in citation_markers:
            if 1 <= marker <= len(context):
                item = context[marker - 1]
                
                if "type" in item and item["type"] == "edge":
                    # Create edge citation
                    citation = {
                        "ref_type": "edge",
                        "ref_id": item.get("id", ""),
                        "relation": item.get("relation", ""),
                        "validity": {
                            "t_valid_start": item.get("t_valid_start", ""),
                            "t_valid_end": item.get("t_valid_end", "")
                        },
                        "source_tool": item.get("source_tool", "")
                    }
                else:
                    # Create chunk citation
                    citation = {
                        "ref_type": "chunk",
                        "ref_id": item.get("chunk_id", ""),
                        "doc_id": item.get("doc_id", ""),
                        "source_tool": item.get("source_tool", ""),
                        "timestamp": item.get("ts_source", "")
                    }
                    
                citations.append(citation)
                
        return response, citations
        
    def _extract_citations_from_context(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation objects from context
        
        Args:
            context: Retrieved context
            
        Returns:
            List of citation objects
        """
        citations = []
        
        for i, item in enumerate(context):
            if "type" in item and item["type"] == "edge":
                # Create edge citation
                citation = {
                    "index": i + 1,
                    "ref_type": "edge",
                    "ref_id": item.get("id", ""),
                    "relation": item.get("relation", ""),
                    "validity": {
                        "t_valid_start": item.get("t_valid_start", ""),
                        "t_valid_end": item.get("t_valid_end", "")
                    },
                    "source_tool": item.get("source_tool", "")
                }
            else:
                # Create chunk citation
                citation = {
                    "index": i + 1,
                    "ref_type": "chunk",
                    "ref_id": item.get("chunk_id", ""),
                    "doc_id": item.get("doc_id", ""),
                    "source_tool": item.get("source_tool", ""),
                    "timestamp": item.get("ts_source", "")
                }
                
            citations.append(citation)
            
        return citations