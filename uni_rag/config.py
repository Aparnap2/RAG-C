from pydantic import BaseModel
from typing import Optional, Dict, Any

class RAGConfig(BaseModel):
    """Configuration for the RAG system"""
    # MCP configuration
    mcp: Optional[Dict[str, Any]] = None  # MCP servers and tenant configuration
    
    # Ingestion configuration
    ingestion: Dict[str, Any]  # e.g., {"pdf": {...}, "web": {...}}
    
    # Storage configuration
    vector_store: Dict[str, Any]  # e.g., {"provider": "astradb", ...}
    text_index: Optional[Dict[str, Any]] = None  # e.g., {"provider": "opensearch", ...}
    knowledge_graph: Optional[Dict[str, Any]] = None  # e.g., {"provider": "neo4j", ...}
    memory: Optional[Dict[str, Any]] = None  # e.g., {"provider": "mem0", ...}
    
    # Retrieval configuration
    retrieval: Optional[Dict[str, Any]] = None  # e.g., {"rrf_k": 60, "vector_weight": 1.0, ...}
    reranker: Optional[Dict[str, Any]] = None  # e.g., {"model_name": "cross-encoder/...", "threshold": 0.8}
    
    # Advanced processing configuration
    chunking: Optional[Dict[str, Any]] = None  # e.g., {"chunk_sizes": [200, 400, 800], "overlap_ratio": 0.1}
    llm_orchestration: Optional[Dict[str, Any]] = None  # e.g., {"llm_tool": "llm.generate", "enable_preprocessing": true}
    
    # Generation configuration
    llm: Dict[str, Any]  # e.g., {"provider": "google_genai", "model": "gemini-1.5-pro"}
    embedding: Optional[Dict[str, Any]] = None  # e.g., {"provider": "google_genai", "api_key": "..."}
    mcp: Optional[Dict[str, Any]] = None  # MCP host configuration
    prompt_template: Optional[str] = None
    
    # Observability configuration
    observability: Optional[Dict[str, Any]] = None  # e.g., {"metrics": {...}, "tracing": {...}} 