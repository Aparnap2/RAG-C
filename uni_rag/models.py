from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class Document(BaseModel):
    """Document model with enhanced fields"""
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    tenant_id: Optional[str] = None
    source_tool: Optional[str] = None
    source_id: Optional[str] = None
    ts_source: Optional[str] = None
    ts_ingested: Optional[str] = None
    acl: List[str] = []
    checksum: Optional[str] = None

class RAGQuery(BaseModel):
    """Legacy query model for backward compatibility"""
    query: str
    source_type: str  # "pdf" or "web"
    source_url: Optional[str] = None  # For web sources
    file_path: Optional[str] = None   # For PDF sources
    max_results: int = 5

class RAGResponse(BaseModel):
    """Legacy response model for backward compatibility"""
    answer: str
    sources: List[Document]
    context: str

class SourceEvent(BaseModel):
    """Event from an external source via MCP"""
    tool_id: str
    data: Dict[str, Any]
    id: Optional[str] = None

class HybridQuery(BaseModel):
    """Query for hybrid retrieval"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    use_graph: bool = False
    top_k: Optional[int] = None
    stream: bool = False

class Citation(BaseModel):
    """Citation for a source"""
    ref_type: str  # "chunk" or "edge"
    ref_id: str
    source_tool: str
    timestamp: Optional[str] = None
    validity: Optional[Dict[str, str]] = None

class HybridResponse(BaseModel):
    """Response from hybrid retrieval"""
    answer: str
    citations: List[Citation]
    has_sufficient_evidence: bool
    evidence_score: float

class ChunkManifestEntry(BaseModel):
    """Entry in a chunk manifest"""
    doc_id: str
    tenant_id: str
    source_tool: str
    source_id: str
    checksum: str
    chunk_count: int
    chunk_ids: List[str]
    ts_created: str
    ts_updated: Optional[str] = None

class Chunk(BaseModel):
    """Chunk of a document"""
    chunk_id: str
    doc_id: str
    text: str
    tokens: int
    tenant_id: str
    source_tool: str
    source_id: str
    acl: List[str]
    metadata: Dict[str, Any]
    ts_source: str
    ts_chunked: str
    chunker_version: str
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None
    ts_embedded: Optional[str] = None

class GraphNode(BaseModel):
    """Node in the knowledge graph"""
    id: str
    type: str
    labels: List[str]
    summary: str
    text: Optional[str] = None
    tenant_id: str
    provenance: Dict[str, Any]

class GraphEdge(BaseModel):
    """Edge in the knowledge graph"""
    id: str
    source_id: str
    target_id: str
    type: str
    t_valid_start: str
    t_valid_end: str
    confidence: float
    tenant_id: str
    provenance: Dict[str, Any]

class EnhancedChunk(BaseModel):
    """Enhanced chunk with multiple representations and metadata."""
    chunk_id: str
    doc_id: str
    text: str
    hypothetical_questions: List[str] = []
    chunk_sizes: Dict[int, str] = {}  # {200: "short_version", 400: "medium_version"}
    metadata_tags: Dict[str, Any] = {}  # product_ID, category, country, etc.
    quality_score: float = 0.5
    embedding_variants: Dict[str, List[float]] = {}  # {task_type: embedding}
    tenant_id: Optional[str] = None
    source_tool: Optional[str] = None
    ts_source: Optional[str] = None
    acl: List[str] = []

class RerankResult(BaseModel):
    """Reranking result with explanation."""
    chunk_id: str
    relevance_score: float
    recency_score: float
    authority_score: float
    combined_score: float
    explanation: Optional[str] = None 