"""Enhanced text sink using LlamaIndex and advanced chunking."""

import asyncio
from typing import List, Dict, Any, Optional
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .models import Document, EnhancedChunk
from .llm_orchestrator import LLMOrchestrator


class EnhancedTextSink:
    """Enhanced text sink with multi-chunk strategies using LlamaIndex."""
    
    def __init__(self, vector_store, text_index, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.text_index = text_index
        self.config = config
        self.chunk_sizes = config.get("chunk_sizes", [200, 400, 800])
        self.overlap_ratio = config.get("overlap_ratio", 0.1)
        
        # Initialize LlamaIndex parsers for different chunk sizes
        self.parsers = {}
        for size in self.chunk_sizes:
            self.parsers[size] = SentenceSplitter(
                chunk_size=size,
                chunk_overlap=int(size * self.overlap_ratio)
            )
        
        # LLM orchestrator for advanced processing
        self.llm_orchestrator = config.get("llm_orchestrator")
        
    async def process_documents(self, documents: List[Document]) -> List[EnhancedChunk]:
        """Process documents with multi-chunk strategies."""
        all_chunks = []
        
        for document in documents:
            # Convert to LlamaIndex document
            llama_doc = LlamaDocument(
                text=document.content,
                metadata={
                    "doc_id": document.id,
                    "tenant_id": document.tenant_id,
                    "source_tool": document.source_tool,
                    "source_id": document.source_id,
                    "ts_source": document.ts_source,
                    "acl": document.acl
                }
            )
            
            # Generate chunks for each size
            doc_chunks = []
            for size in self.chunk_sizes:
                nodes = self.parsers[size].get_nodes_from_documents([llama_doc])
                
                for i, node in enumerate(nodes):
                    chunk = EnhancedChunk(
                        chunk_id=f"{document.id}_{size}_{i}",
                        doc_id=document.id,
                        text=node.text,
                        chunk_sizes={size: node.text},
                        tenant_id=document.tenant_id,
                        source_tool=document.source_tool,
                        ts_source=document.ts_source,
                        acl=document.acl or []
                    )
                    doc_chunks.append(chunk)
            
            # Enhanced processing if LLM orchestrator available
            if self.llm_orchestrator:
                doc_chunks = await self._enhance_chunks(doc_chunks)
            
            all_chunks.extend(doc_chunks)
        
        # Store in vector store and text index
        await self._store_chunks(all_chunks)
        
        return all_chunks
    
    async def _enhance_chunks(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Enhance chunks with hypothetical questions and metadata."""
        enhanced_chunks = []
        
        for chunk in chunks:
            # Generate hypothetical questions
            if self.llm_orchestrator:
                try:
                    questions = await self.llm_orchestrator.generate_hypothetical_questions(chunk)
                    chunk.hypothetical_questions = questions
                    
                    # Enrich metadata
                    metadata = await self.llm_orchestrator.enrich_metadata(chunk)
                    chunk.metadata_tags = metadata
                    
                    # Calculate quality score
                    quality_score = await self.llm_orchestrator.calculate_quality_score(chunk)
                    chunk.quality_score = quality_score
                    
                except Exception as e:
                    # Continue without enhancement if LLM fails
                    pass
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    async def _store_chunks(self, chunks: List[EnhancedChunk]):
        """Store chunks in vector store and text index."""
        if not chunks:
            return
            
        # Convert to Document format for storage
        documents = []
        for chunk in chunks:
            doc = Document(
                id=chunk.chunk_id,
                content=chunk.text,
                metadata={
                    "doc_id": chunk.doc_id,
                    "chunk_sizes": chunk.chunk_sizes,
                    "hypothetical_questions": chunk.hypothetical_questions,
                    "metadata_tags": chunk.metadata_tags,
                    "quality_score": chunk.quality_score
                },
                tenant_id=chunk.tenant_id,
                source_tool=chunk.source_tool,
                ts_source=chunk.ts_source,
                acl=chunk.acl
            )
            documents.append(doc)
        
        # Store in parallel
        tasks = []
        if self.vector_store:
            tasks.append(self.vector_store.add_documents(documents))
        if self.text_index:
            tasks.append(self.text_index.add_documents(documents))
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)