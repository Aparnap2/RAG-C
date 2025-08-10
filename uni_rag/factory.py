from .vector_stores import QdrantVectorStore, AstraVectorStore
from .text_indexes import OpenSearchTextIndex, ElasticsearchTextIndex
from .ingestion import get_ingestion_instance
from .knowledge_graph import KnowledgeGraph
from .memory import ConversationMemory
from .interfaces import LLMBase, VectorStoreBase, KnowledgeGraphBase, MemoryBase, TextIndexBase
from .embeddings import GoogleGenAIEmbedding
from typing import Any, Dict

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

# Placeholder for OpenRouter LLM
class OpenRouterLLM(LLMBase):
    def __init__(self, config: dict):
        # Implementation for OpenRouter would go here
        self.config = config
    def invoke(self, prompt: str) -> Any:
        # Actual call to OpenRouter API
        return "Response from OpenRouter"

# Extend these as you add more implementations

def get_vector_store(config: Dict[str, Any]) -> VectorStoreBase:
    """Factory function to get a vector store instance based on config."""
    provider = config.get("provider")
    if provider == "qdrant":
        return QdrantVectorStore(config)
    elif provider == "astradb":
        return AstraVectorStore(config)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")

def get_text_index(config: Dict[str, Any]) -> TextIndexBase:
    """Factory function to get a text index instance based on config."""
    provider = config.get("provider")
    if provider == "opensearch":
        return OpenSearchTextIndex(config)
    elif provider == "elasticsearch":
        return ElasticsearchTextIndex(config)
    else:
        raise ValueError(f"Unknown text index provider: {provider}")

def get_ingestion(source_type: str, config: dict) -> Any:
    return get_ingestion_instance(source_type).ingest

def get_knowledge_graph(config: dict) -> KnowledgeGraphBase:
    """Factory function to get a knowledge graph instance based on config."""
    provider = config.get("provider")
    if provider == "neo4j":
        return KnowledgeGraph()
    elif provider == "in-memory":
        # Example of how to add another provider
        # from rag.knowledge_graphs.in_memory import InMemoryKnowledgeGraph
        # return InMemoryKnowledgeGraph(config)
        raise NotImplementedError("In-memory knowledge graph not yet implemented.")
    else:
        raise ValueError(f"Unknown knowledge graph provider: {provider}")

def get_memory(config: dict) -> MemoryBase:
    return ConversationMemory()

def get_llm(config: Dict[str, Any]) -> LLMBase:
    """Factory function to get an LLM instance based on config."""
    provider = config.get("provider")
    if provider == "google_genai":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai not installed. Install with: pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=config.get("model", "gemini-1.5-pro"),
            google_api_key=config.get("google_api_key")
        )
    elif provider == "openrouter":
        return OpenRouterLLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def get_embedding_client(config: Dict[str, Any]):
    """Factory function to get an embedding client based on config."""
    provider = config.get("provider", "google_genai")
    if provider == "google_genai":
        return GoogleGenAIEmbedding(config["api_key"])
    else:
        raise ValueError(f"Unknown embedding provider: {provider}") 