from pydantic import BaseModel
from typing import Optional, Dict, Any

class RAGConfig(BaseModel):
    ingestion: Dict[str, Any]  # e.g., {"pdf": {...}, "web": {...}}
    vector_store: Dict[str, Any]  # e.g., {"provider": "qdrant", ...}
    knowledge_graph: Optional[Dict[str, Any]]
    memory: Optional[Dict[str, Any]]
    llm: Dict[str, Any]  # e.g., {"provider": "openai", "model": "gpt-4"}
    prompt_template: Optional[str] 