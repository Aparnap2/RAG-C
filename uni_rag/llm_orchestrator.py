"""Multi-LLM orchestration for advanced RAG techniques."""

import asyncio
from typing import List, Dict, Any, Optional
from .models import EnhancedChunk


class LLMOrchestrator:
    """Orchestrates multiple LLM calls for advanced RAG techniques."""
    
    def __init__(self, mcp_host, config: Dict[str, Any]):
        self.mcp_host = mcp_host
        self.config = config
        self.llm_tool = config.get("llm_tool", "llm.generate")
        
    async def preprocess_query(self, query: str) -> str:
        """Preprocess query: spelling correction, expansion, synonym replacement."""
        prompt = f"""
        Improve this query for better search results:
        - Fix spelling errors
        - Expand abbreviations
        - Add relevant synonyms
        - Keep the original intent
        
        Query: {query}
        
        Improved query:"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 100}
            )
            return result.get("text", query).strip()
        except Exception:
            return query
    
    async def generate_hypothetical_questions(self, chunk: EnhancedChunk) -> List[str]:
        """Generate hypothetical questions this chunk could answer."""
        prompt = f"""
        Generate 3-5 specific questions that this text chunk could answer well.
        Make questions diverse and specific to the content.
        
        Text: {chunk.text[:500]}...
        
        Questions (one per line):"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 200}
            )
            questions = result.get("text", "").strip().split('\n')
            return [q.strip('- ').strip() for q in questions if q.strip()]
        except Exception:
            return []
    
    async def enrich_metadata(self, chunk: EnhancedChunk) -> Dict[str, Any]:
        """Generate metadata tags for chunk."""
        prompt = f"""
        Analyze this text and extract metadata tags:
        - topic: main topic/subject
        - category: content category
        - entities: key entities mentioned
        - sentiment: positive/negative/neutral
        
        Text: {chunk.text[:300]}...
        
        Return as JSON:"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 150}
            )
            # Parse JSON response
            import json
            metadata = json.loads(result.get("text", "{}"))
            return metadata
        except Exception:
            return {}
    
    async def summarize_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Summarize retrieved chunks before generation."""
        if not chunks:
            return ""
            
        context_text = "\n\n".join([
            f"Source {i+1}: {chunk.get('text', '')[:200]}..."
            for i, chunk in enumerate(chunks[:5])
        ])
        
        prompt = f"""
        Summarize the key information from these sources:
        
        {context_text}
        
        Summary:"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 300}
            )
            return result.get("text", "").strip()
        except Exception:
            return context_text
    
    async def evaluate_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """Self-evaluate response accuracy and relevance."""
        prompt = f"""
        Evaluate this response for accuracy and relevance:
        
        Query: {query}
        Response: {response}
        Context: {context[:500]}...
        
        Rate on scale 1-10:
        - Accuracy: How factually correct is the response?
        - Relevance: How well does it answer the query?
        - Completeness: How complete is the answer?
        
        Return as JSON with scores and brief explanation:"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 200}
            )
            import json
            evaluation = json.loads(result.get("text", "{}"))
            return evaluation
        except Exception:
            return {"accuracy": 5, "relevance": 5, "completeness": 5}
    
    async def calculate_quality_score(self, chunk: EnhancedChunk, 
                                    source_authority: float = 0.5) -> float:
        """Calculate quality score for chunk."""
        # Base score from source authority
        score = source_authority
        
        # Adjust based on content length (not too short, not too long)
        text_length = len(chunk.text.split())
        if 50 <= text_length <= 300:
            score += 0.2
        elif text_length < 20:
            score -= 0.3
            
        # Adjust based on metadata richness
        if chunk.metadata_tags:
            score += 0.1 * min(len(chunk.metadata_tags), 3)
            
        # Adjust based on hypothetical questions
        if chunk.hypothetical_questions:
            score += 0.1 * min(len(chunk.hypothetical_questions), 2)
            
        return max(0.0, min(1.0, score))