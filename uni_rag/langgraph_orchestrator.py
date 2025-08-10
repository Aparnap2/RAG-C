"""LangGraph-based orchestration for advanced RAG workflows."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from .models import EnhancedChunk, RerankResult


class RAGState(TypedDict):
    """State for RAG workflow."""
    query: str
    processed_query: str
    candidates: List[Dict[str, Any]]
    reranked: List[RerankResult]
    context: str
    response: str
    evaluation: Dict[str, Any]
    metadata: Dict[str, Any]


class LangGraphOrchestrator:
    """LangGraph-based orchestration for multi-LLM RAG workflow."""
    
    def __init__(self, mcp_host, retriever, reranker, generator, config: Dict[str, Any]):
        self.mcp_host = mcp_host
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.config = config
        self.llm_tool = config.get("llm_tool", "llm.generate")
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the RAG workflow graph."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("preprocess_query", self._preprocess_query)
        workflow.add_node("retrieve_candidates", self._retrieve_candidates)
        workflow.add_node("rerank_results", self._rerank_results)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("evaluate_response", self._evaluate_response)
        
        # Add edges
        workflow.set_entry_point("preprocess_query")
        workflow.add_edge("preprocess_query", "retrieve_candidates")
        workflow.add_edge("retrieve_candidates", "rerank_results")
        workflow.add_edge("rerank_results", "generate_response")
        workflow.add_edge("generate_response", "evaluate_response")
        workflow.add_edge("evaluate_response", END)
        
        return workflow.compile()
    
    async def _preprocess_query(self, state: RAGState) -> RAGState:
        """Preprocess query for better retrieval."""
        query = state["query"]
        
        prompt = f"""
        Improve this query for better search results:
        - Fix spelling errors
        - Expand abbreviations  
        - Add relevant synonyms
        - Keep original intent
        
        Query: {query}
        Improved query:"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 100}
            )
            processed_query = result.get("text", query).strip()
        except Exception:
            processed_query = query
            
        state["processed_query"] = processed_query
        return state
    
    async def _retrieve_candidates(self, state: RAGState) -> RAGState:
        """Retrieve candidate documents."""
        query = state["processed_query"]
        
        # Use hybrid retrieval
        filters = state.get("metadata", {}).get("filters", {})
        candidates = await self.retriever.retrieve(
            query, 
            filters, 
            top_k=self.config.get("retrieval_k", 50)
        )
        
        state["candidates"] = candidates
        return state
    
    async def _rerank_results(self, state: RAGState) -> RAGState:
        """Rerank candidates with cross-encoder."""
        query = state["processed_query"]
        candidates = state["candidates"]
        
        if not candidates:
            state["reranked"] = []
            return state
        
        # Extract features for reranking
        features = await self.reranker.extract_features(query, candidates)
        
        # Rerank with quality threshold
        reranked = await self.reranker.rerank(
            query,
            candidates,
            features,
            top_k=self.config.get("final_k", 5)
        )
        
        # Filter by quality threshold
        threshold = self.config.get("quality_threshold", 0.7)
        filtered = [r for r in reranked if r.get("score", 0) >= threshold]
        
        state["reranked"] = filtered
        return state
    
    async def _generate_response(self, state: RAGState) -> RAGState:
        """Generate grounded response."""
        query = state["query"]
        reranked = state["reranked"]
        
        if not reranked:
            state["response"] = "I don't have enough relevant information to answer that question."
            state["context"] = ""
            return state
        
        # Summarize context
        context_chunks = [r.get("text", "") for r in reranked[:3]]
        context = await self._summarize_context(context_chunks)
        
        # Generate response
        response = await self.generator.generate_with_citations(
            query, reranked
        )
        
        state["context"] = context
        state["response"] = response.get("answer", "")
        return state
    
    async def _evaluate_response(self, state: RAGState) -> RAGState:
        """Self-evaluate response quality."""
        query = state["query"]
        response = state["response"]
        context = state["context"]
        
        prompt = f"""
        Evaluate this response:
        Query: {query}
        Response: {response}
        Context: {context[:300]}...
        
        Rate 1-10:
        - Accuracy: factual correctness
        - Relevance: answers the query
        - Completeness: thorough answer
        
        JSON format: {{"accuracy": X, "relevance": Y, "completeness": Z}}"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 100}
            )
            import json
            evaluation = json.loads(result.get("text", "{}"))
        except Exception:
            evaluation = {"accuracy": 5, "relevance": 5, "completeness": 5}
        
        state["evaluation"] = evaluation
        return state
    
    async def _summarize_context(self, chunks: List[str]) -> str:
        """Summarize context chunks."""
        if not chunks:
            return ""
            
        context_text = "\n\n".join(chunks[:3])
        
        prompt = f"""
        Summarize key information:
        {context_text}
        
        Summary:"""
        
        try:
            result = await self.mcp_host.invoke_tool(
                self.llm_tool,
                {"prompt": prompt, "max_tokens": 200}
            )
            return result.get("text", "").strip()
        except Exception:
            return context_text
    
    async def process_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process query through the workflow."""
        initial_state = RAGState(
            query=query,
            processed_query="",
            candidates=[],
            reranked=[],
            context="",
            response="",
            evaluation={},
            metadata=metadata or {}
        )
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        return {
            "answer": final_state["response"],
            "context": final_state["context"],
            "evaluation": final_state["evaluation"],
            "candidates_count": len(final_state["candidates"]),
            "reranked_count": len(final_state["reranked"])
        }