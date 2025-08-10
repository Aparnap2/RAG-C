from .models import RAGQuery, RAGResponse, Document
from .config import RAGConfig
from .factory import get_vector_store, get_text_index, get_knowledge_graph, get_memory, get_llm, get_embedding_client
from .retrieval_hybrid import HybridRetriever
from .reranker import CrossEncoderReranker
from .grounding import GroundedGenerator
from .enhanced_text_sink import EnhancedTextSink
from .langgraph_orchestrator import LangGraphOrchestrator
from .unified_ingestion import UnifiedIngestion
from .mcp.host import MCPHost
import os
import asyncio
from typing import Optional, List, Union, Dict, Any

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        # Initialize MCP host if configured
        self.mcp_host = None
        if config.mcp:
            self.mcp_host = MCPHost(config.mcp)
            
        # Initialize embedding client
        embedding_client = None
        if config.embedding:
            embedding_client = get_embedding_client(config.embedding)
            
        # Initialize vector store with embedding client
        vector_config = config.vector_store.copy()
        if embedding_client:
            vector_config["embedding_client"] = embedding_client
        self.vector_store = get_vector_store(vector_config)
        
        # Initialize text index
        self.text_index = get_text_index(config.text_index) if config.text_index else None
        
        # Initialize enhanced text sink
        sink_config = {
            "chunk_sizes": config.chunking.get("chunk_sizes", [200, 400, 800]) if config.chunking else [200, 400, 800],
            "overlap_ratio": config.chunking.get("overlap_ratio", 0.1) if config.chunking else 0.1,
            "llm_orchestrator": self.mcp_host
        }
        self.text_sink = EnhancedTextSink(self.vector_store, self.text_index, sink_config)
        
        # Initialize unified ingestion
        self.ingestion = UnifiedIngestion(config.ingestion or {})
        self.llm = get_llm(config.llm)
        self.memory = get_memory(config.memory) if config.memory else None
        self.knowledge_graph = get_knowledge_graph(config.knowledge_graph) if config.knowledge_graph else None
        
        # Initialize retrieval components
        if self.text_index:
            self.hybrid_retriever = HybridRetriever(
                self.vector_store, 
                self.text_index, 
                config.retrieval or {}
            )
        else:
            self.hybrid_retriever = None
            
        # Initialize reranker and generator
        self.reranker = CrossEncoderReranker(
            self.mcp_host, 
            None,  # cache_client
            config.reranker or {}
        )
        self.grounded_generator = GroundedGenerator(
            self.mcp_host,
            config.llm
        )
        
        # Initialize LangGraph orchestrator if configured
        self.orchestrator = None
        if config.llm_orchestration and self.mcp_host:
            self.orchestrator = LangGraphOrchestrator(
                self.mcp_host,
                self.hybrid_retriever,
                self.reranker,
                self.grounded_generator,
                config.llm_orchestration
            )
        
        self.prompt_template = config.prompt_template or "Context: {context}\nQuery: {query}\nAnswer concisely:"

    async def ingest(self, source: Union[str, List[str]], 
                   source_type: str = "auto",
                   metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Unified ingestion method."""
        documents = await self.ingestion.ingest(source, source_type, metadata)
        
        # Process with enhanced text sink
        if documents:
            chunks = await self.text_sink.process_documents(documents)
            
            # Add to knowledge graph if available
            if self.knowledge_graph:
                for doc in documents:
                    await self.knowledge_graph.add_document(doc)
        
        return documents
    
    async def ingest_mixed_sources(self, sources: Dict[str, List[str]], 
                                 metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest from mixed source types."""
        return await self.ingestion.ingest_mixed_sources(sources, metadata)

    async def query(self, query: RAGQuery) -> RAGResponse:
        """Query using advanced RAG pipeline."""
        # Ingest new data if provided
        if query.file_path:
            await self.ingest(query.file_path, "file")
        elif query.source_url:
            await self.ingest(query.source_url, "url")

        # Use LangGraph orchestrator if available
        if self.orchestrator:
            result = await self.orchestrator.process_query(
                query.query,
                metadata={"filters": {"tenant_id": getattr(query, 'tenant_id', 'default')}}
            )
            
            # Convert to legacy format
            return RAGResponse(
                answer=result["answer"],
                sources=[],  # Sources embedded in context
                context=result["context"]
            )
        
        # Fallback to basic pipeline
        return await self._basic_query(query)
    
    async def _basic_query(self, query: RAGQuery) -> RAGResponse:
        """Basic query pipeline for backward compatibility."""
        # Retrieve context from memory
        memory_context = await self.memory.get_context(query.query) if self.memory else ""

        # Use hybrid retrieval if available
        if self.hybrid_retriever:
            filters = {"tenant_id": getattr(query, 'tenant_id', 'default')}
            results = await self.hybrid_retriever.retrieve(
                query.query, 
                filters, 
                top_k=query.max_results
            )
            # Convert to Document objects
            vector_results = []
            for result in results:
                doc = Document(
                    id=result["id"],
                    content=result["content"],
                    metadata=result.get("metadata", {})
                )
                vector_results.append(doc)
        else:
            vector_results = await self.vector_store.search(query.query, query.max_results)

        # Combine contexts
        context = memory_context + "\n" + "\n".join([doc.content for doc in vector_results])

        # Generate response
        prompt = self.prompt_template.format(context=context, query=query.query)
        response = self.llm.invoke(prompt).content

        return RAGResponse(
            answer=response,
            sources=vector_results,
            context=context
        ) 