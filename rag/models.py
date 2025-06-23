from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: str
    content: str
    metadata: dict = {}

class RAGQuery(BaseModel):
    query: str
    source_type: str  # "pdf" or "web"
    source_url: Optional[str] = None  # For web sources
    file_path: Optional[str] = None   # For PDF sources
    max_results: int = 5

class RAGResponse(BaseModel):
    answer: str
    sources: List[Document]
    context: str 