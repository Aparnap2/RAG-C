import os
import json
import time
import shutil
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from uni_rag.config import RAGConfig
from uni_rag.models import RAGQuery, RAGResponse, Document, SourceEvent, HybridQuery
from uni_rag.mcp.host import MCPHost
from uni_rag.retrieval_hybrid import HybridRetriever
from uni_rag.reranker import CrossEncoderReranker
from uni_rag.grounding import GroundedGenerator
from uni_rag.enhanced_text_sink import EnhancedTextSink
from api.observability import RAGObservability

app = FastAPI(title="MCP-First, Type-C RAG Platform")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config from file or environment variable
CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "rag_config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config_dict = json.load(f)
    config = RAGConfig(**config_dict)
else:
    config = RAGConfig(
        mcp={
            "servers": {},
            "tenants": {}
        },
        ingestion={
            "pdf": {},
            "web": {}
        },
        vector_store={
            "provider": "astradb"
        },
        text_index={
            "provider": "opensearch"
        },
        knowledge_graph={
            "provider": "neo4j"
        },
        retrieval={
            "rrf_k": 60,
            "vector_weight": 1.0,
            "bm25_weight": 1.0
        },
        reranker={
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "recency_weight": 0.1,
            "entity_weight": 0.2
        },
        llm={
            "provider": "google_genai",
            "model": "gemini-1.5-pro"
        },
        prompt_template=None
    )

# Initialize components
# Note: In a real implementation, these would be actual clients
# For this example, we'll use mock clients

class MockVectorStore:
    async def search(self, query, top_k=10):
        return []
        
    async def get_documents(self, doc_ids):
        return []
        
    async def upsert(self, documents):
        pass
        
    async def delete(self, doc_ids):
        pass
        
    async def health_check(self):
        return True

class MockTextIndex:
    async def search(self, query, top_k=10):
        return []
        
    async def upsert(self, documents):
        pass
        
    async def delete(self, doc_ids):
        pass
        
    async def health_check(self):
        return True

class MockGraphClient:
    async def get_node(self, node_id):
        return None
        
    async def create_node(self, node_id, node_type, properties):
        pass
        
    async def update_node(self, node_id, properties):
        pass
        
    async def get_edges(self, source_id, target_id, edge_type):
        return []
        
    async def create_edge(self, edge_id, source_id, target_id, edge_type, properties):
        pass
        
    async def update_edge(self, edge_id, properties):
        pass
        
    async def health_check(self):
        return True

class MockEmbeddingClient:
    async def embed_documents(self, texts, model):
        return [[0.0] * 768 for _ in texts]

class MockLLMClient:
    async def generate(self, prompt):
        return f"Response to: {prompt[:50]}..."
        
    async def generate_stream(self, prompt):
        yield f"Response to: {prompt[:50]}..."

class MockCacheClient:
    async def get(self, key):
        return None
        
    async def set(self, key, value, ttl=None):
        pass

# Initialize components
vector_store = MockVectorStore()
text_index = MockTextIndex()
graph_client = MockGraphClient()
embedding_client = MockEmbeddingClient()
llm_client = MockLLMClient()
cache_client = MockCacheClient()

# Initialize MCP host
mcp_host = MCPHost(config.mcp)

# Initialize queue client
queue_client = InMemoryQueueClient()

# Initialize workers
ingestion_worker = MCPIngestionWorker(mcp_host, queue_client, config.ingestion)
normalizer = Normalizer(config.ingestion)
graph_sink = GraphSink(graph_client, config.knowledge_graph)
text_sink = TextSink(vector_store, text_index, {
    "chunker": config.ingestion,
    "embedding_client": embedding_client,
    "embedder": config.retrieval
})

# Initialize retrieval components
hybrid_retriever = HybridRetriever(vector_store, text_index, config.retrieval)
reranker = CrossEncoderReranker(llm_client, cache_client, config.reranker)
grounded_generator = GroundedGenerator(llm_client, config.llm)

# Initialize observability
observability = RAGObservability(config)

# Dependency for tenant validation
async def validate_tenant(tenant_id: str = Header(...)):
    # In a real implementation, this would validate the tenant
    # For now, just return the tenant ID
    return tenant_id

@app.post("/ingest/events")
async def ingest_events(
    event: SourceEvent,
    tenant_id: str = Depends(validate_tenant),
    user_id: Optional[str] = Header(None)
):
    # Start trace
    with observability.start_trace(
        "api.ingest_events",
        attributes={"tenant_id": tenant_id, "tool_id": event.tool_id}
    ):
        # Process event
        try:
            start_time = time.time()
            
            result = await ingestion_worker.process_event(
                event.tool_id,
                event.data,
                tenant_id,
                user_id
            )
            
            latency = time.time() - start_time
            await observability.trace_ingestion(
                event.tool_id,
                1,
                result.get("items_processed", 0),
                latency
            )
            
            return {
                "status": "success",
                "document_id": result.get("document_id")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    tenant_id: str = Depends(validate_tenant),
    user_id: Optional[str] = Header(None)
):
    # Save file temporarily
    file_path = f"tmp/{file.filename}"
    os.makedirs("tmp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Start trace
        with observability.start_trace(
            "api.ingest_file",
            attributes={"tenant_id": tenant_id, "filename": file.filename}
        ):
            start_time = time.time()
            
            # Create document
            document = {
                "tenant_id": tenant_id,
                "source_tool": "file_upload",
                "source_id": file.filename,
                "content": "File content will be extracted",
                "metadata": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": file.size
                },
                "acl": [f"tenant:{tenant_id}"],
                "ts_source": datetime.now().isoformat(),
                "ts_ingested": datetime.now().isoformat()
            }
            
            # Process document
            result = await ingestion_worker.process_event(
                "file_upload",
                document,
                tenant_id,
                user_id
            )
            
            latency = time.time() - start_time
            await observability.trace_ingestion(
                "file_upload",
                1,
                result.get("items_processed", 0),
                latency
            )
            
            return {
                "status": "success",
                "document_id": result.get("document_id")
            }
    finally:
        # Clean up
        os.remove(file_path)

@app.post("/query")
async def query(
    query: HybridQuery,
    response: Response,
    tenant_id: str = Depends(validate_tenant),
    user_id: Optional[str] = Header(None)
):
    # Start trace
    with observability.start_trace(
        "api.query",
        attributes={"tenant_id": tenant_id, "query": query.query}
    ):
        try:
            # Apply filters
            filters = {
                "tenant_id": tenant_id,
                "acl": [f"tenant:{tenant_id}"]
            }
            
            if query.filters:
                filters.update(query.filters)
            
            # Retrieve candidates
            start_time = time.time()
            
            if query.use_graph:
                candidates = await hybrid_retriever.retrieve_with_graph(
                    query.query,
                    filters,
                    graph_client,
                    top_k=query.top_k or 50
                )
            else:
                candidates = await hybrid_retriever.retrieve(
                    query.query,
                    filters,
                    top_k=query.top_k or 50
                )
                
            retrieval_latency = time.time() - start_time
            await observability.trace_retrieval(
                query.query,
                filters,
                candidates,
                retrieval_latency
            )
            
            # Rerank candidates
            start_time = time.time()
            
            # Extract features
            features = await reranker.extract_features(
                query.query,
                candidates,
                graph_client if query.use_graph else None
            )
            
            reranked = await reranker.rerank(
                query.query,
                candidates,
                features,
                top_k=query.top_k or 5
            )
            
            reranking_latency = time.time() - start_time
            await observability.trace_reranking(
                query.query,
                candidates,
                reranked,
                reranking_latency
            )
            
            # Generate grounded response
            start_time = time.time()
            
            if query.stream:
                # Set up SSE streaming
                response.headers["Content-Type"] = "text/event-stream"
                return EventSourceResponse(
                    grounded_generator.generate_stream(
                        query.query,
                        reranked,
                        tenant_id
                    )
                )
            else:
                # Generate complete response
                result = await grounded_generator.generate_with_citations(
                    query.query,
                    reranked,
                    tenant_id
                )
                
                generation_latency = time.time() - start_time
                await observability.trace_generation(
                    query.query,
                    sum(len(doc.get("text", "")) for doc in reranked),
                    len(result.get("answer", "")),
                    generation_latency
                )
                
                return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return await observability.get_metrics()

@app.get("/healthz")
async def health_check():
    # Check all components
    status = {
        "vector_store": await vector_store.health_check(),
        "text_index": await text_index.health_check(),
        "knowledge_graph": await graph_client.health_check(),
        "mcp_host": await mcp_host.health_check()
    }
    
    return {
        "status": "healthy" if all(status.values()) else "unhealthy",
        "components": status
    }

# Legacy endpoints for backward compatibility
@app.post("/api/rag/query")
async def legacy_query_rag(
    query: RAGQuery,
    tenant_id: str = Header("default"),
    user_id: Optional[str] = Header(None)
):
    # Convert to new query format
    hybrid_query = HybridQuery(
        query=query.query,
        filters={},
        use_graph=False,
        top_k=query.max_results,
        stream=False
    )
    
    # Call new endpoint
    return await query(hybrid_query, Response(), tenant_id, user_id)

@app.post("/api/rag/ingest")
async def legacy_ingest_file(
    file: UploadFile = File(...),
    tenant_id: str = Header("default"),
    user_id: Optional[str] = Header(None)
):
    # Call new endpoint
    return await ingest_file(file, tenant_id, user_id) 