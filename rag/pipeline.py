from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from rag.models import RAGQuery, RAGResponse, Document
from rag.vector_store import VectorStore
from rag.knowledge_graph import KnowledgeGraph
from rag.memory import ConversationMemory
from rag.ingestion import ingest_document
from rag.config import RAGConfig
from rag.factory import get_vector_store, get_ingestion, get_knowledge_graph, get_memory, get_llm
import os
import asyncio
from typing import Optional

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = get_vector_store(config.vector_store)
        self.llm = get_llm(config.llm)
        self.memory = get_memory(config.memory) if config.memory else None
        self.knowledge_graph = get_knowledge_graph(config.knowledge_graph) if config.knowledge_graph else None
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

        # Search vector store
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