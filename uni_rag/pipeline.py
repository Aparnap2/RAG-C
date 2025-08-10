from .models import RAGQuery, RAGResponse, Document
from .config import RAGConfig
from .factory import get_vector_store, get_text_index, get_ingestion, get_knowledge_graph, get_memory, get_llm, get_embedding_client
from .retrieval_hybrid import HybridRetriever
from .reranker import CrossEncoderReranker
from .grounding import GroundedGenerator
import os
import asyncio
from typing import Optional

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
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
            
        # Initialize reranker and generator (mock implementations for now)
        self.reranker = None  # Will be initialized with real model client
        self.grounded_generator = None  # Will be initialized with real LLM client
        
        self.prompt_template = config.prompt_template or "Context: {context}\nQuery: {query}\nAnswer concisely:"

    async def ingest(self, source_type: str, **kwargs):
        ingestion_fn = get_ingestion(source_type, self.config.ingestion.get(source_type, {}))
        documents = await ingestion_fn(**kwargs)
        await self.vector_store.add_documents(documents)
        if self.knowledge_graph:
            for doc in documents:
                await self.knowledge_graph.add_document(doc)
        return documents

    async def query(self, query: RAGQuery) -> RAGResponse:
        # Ingest new data if provided
        if query.source_url or query.file_path:
            await self.ingest(query.source_type, source_url=query.source_url, file_path=query.file_path)

        # Retrieve context from memory
        memory_context = await self.memory.get_context(query.query) if self.memory else ""

        # Use hybrid retrieval if available, otherwise fallback to vector only
        if self.hybrid_retriever:
            # Hybrid search with RRF
            filters = {"tenant_id": getattr(query, 'tenant_id', 'default')}
            results = await self.hybrid_retriever.retrieve(
                query.query, 
                filters, 
                top_k=query.max_results
            )
            # Convert dict results back to Document objects for compatibility
            vector_results = []
            for result in results:
                doc = Document(
                    id=result["id"],
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    tenant_id=result.get("tenant_id"),
                    source_tool=result.get("source_tool"),
                    source_id=result.get("source_id"),
                    ts_source=result.get("ts_source"),
                    ts_ingested=result.get("ts_ingested"),
                    acl=result.get("acl", [])
                )
                vector_results.append(doc)
        else:
            # Fallback to vector-only search
            vector_results = await self.vector_store.search(query.query, query.max_results)

        # Query knowledge graph
        graph_context = ""
        if self.knowledge_graph:
            graph_results = await self.knowledge_graph.query_relations(query.query)
            graph_context = "\n".join([res["properties"].get("content", "") for res in graph_results])

        # Combine contexts
        context = f"{memory_context}\n{graph_context}\n" + "\n".join([doc.content for doc in vector_results])

        # Use dynamic prompt template
        prompt = self.prompt_template.format(context=context, query=query.query)
        response = self.llm.invoke(prompt).content

        # Store response in memory
        if self.memory:
            await self.memory.add_context(query, response)

        return RAGResponse(
            answer=response,
            sources=vector_results,
            context=context
        ) 