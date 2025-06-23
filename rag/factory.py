from rag.vector_store import VectorStore
from rag.ingestion import get_ingestion_instance
from rag.knowledge_graph import KnowledgeGraph
from rag.memory import ConversationMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Any

# Extend these as you add more implementations

def get_vector_store(config: dict) -> Any:
    return VectorStore(provider=config.get("provider", "astradb"))

def get_ingestion(source_type: str, config: dict) -> Any:
    return get_ingestion_instance(source_type).ingest

def get_knowledge_graph(config: dict) -> Any:
    return KnowledgeGraph()

def get_memory(config: dict) -> Any:
    return ConversationMemory()

def get_llm(config: dict) -> Any:
    provider = config.get("provider", "google_genai")
    if provider == "google_genai":
        return ChatGoogleGenerativeAI(
            model=config.get("model", "gemini-1.5-pro"),
            google_api_key=config.get("google_api_key")
        )
    raise ValueError(f"Unknown LLM provider: {provider}") 