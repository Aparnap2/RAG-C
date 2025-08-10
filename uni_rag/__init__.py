"""
RAG-C: MCP-First, Type-C RAG Platform

A universal RAG engine with pluggable components for vector stores,
knowledge graphs, and data sources via MCP protocol.
"""

__version__ = "0.1.0"

from .config import RAGConfig
from .models import Document, RAGQuery, RAGResponse
from .pipeline import RAGPipeline
from .factory import (
    get_vector_store,
    get_knowledge_graph, 
    get_memory,
    get_llm,
    get_embedding_client
)
from .llm_orchestrator import LLMOrchestrator
from .unified_ingestion import UnifiedIngestion

__all__ = [
    "RAGConfig",
    "Document", 
    "RAGQuery",
    "RAGResponse",
    "RAGPipeline",
    "get_vector_store",
    "get_knowledge_graph",
    "get_memory", 
    "get_llm",
    "get_embedding_client",
    "LLMOrchestrator",
    "UnifiedIngestion",
]