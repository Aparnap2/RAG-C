"""Advanced usage example with LangGraph orchestration."""

import asyncio
from uni_rag import RAGPipeline, RAGConfig, RAGQuery

async def main():
    # Advanced configuration with LangGraph orchestration
    config = RAGConfig(
        # MCP configuration for LLM calls
        mcp={
            "servers": {
                "llm_server": {
                    "transport": "stdio",
                    "command": "python -m llm_mcp_server"
                }
            },
            "tenants": {
                "demo": {"allowed_tools": ["llm.generate"]}
            }
        },
        
        # Vector store with embeddings
        vector_store={
            "provider": "qdrant",
            "url": "http://localhost:6333",
            "collection_name": "advanced_docs",
            "vector_size": 768
        },
        
        # Text index for hybrid search
        text_index={
            "provider": "opensearch",
            "host": "localhost",
            "port": 9200,
            "index_name": "advanced_docs"
        },
        
        # Google GenAI embeddings
        embedding={
            "provider": "google_genai",
            "api_key": "your-google-api-key"
        },
        
        # LLM configuration
        llm={
            "provider": "google_genai",
            "model": "gemini-1.5-pro",
            "google_api_key": "your-google-api-key"
        },
        
        # Advanced chunking with multiple sizes
        chunking={
            "chunk_sizes": [200, 400, 800],
            "overlap_ratio": 0.15
        },
        
        # LangGraph orchestration
        llm_orchestration={
            "llm_tool": "llm.generate",
            "enable_preprocessing": True,
            "retrieval_k": 50,
            "final_k": 5,
            "quality_threshold": 0.7
        },
        
        # Enhanced reranking
        reranker={
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "threshold": 0.8,
            "recency_weight": 0.1,
            "entity_weight": 0.2
        },
        
        # Basic ingestion config
        ingestion={
            "use_docling": True,
            "max_concurrent": 5
        }
    )
    
    # Create advanced pipeline
    pipeline = RAGPipeline(config)
    
    # Ingest documents using Docling
    print("Ingesting documents...")
    documents = await pipeline.ingest_directory(
        "path/to/documents",
        file_patterns=["*.pdf", "*.docx", "*.txt"],
        metadata={"source": "demo", "category": "technical"}
    )
    print(f"Ingested {len(documents)} documents")
    
    # Advanced query with LangGraph orchestration
    print("\nProcessing query with advanced RAG...")
    query = RAGQuery(
        query="What are the key benefits of artificial intelligence?",
        max_results=5,
        source_type="file"
    )
    
    response = await pipeline.query(query)
    
    print(f"Answer: {response.answer}")
    print(f"Context length: {len(response.context)} characters")
    
    # Example of direct file ingestion
    print("\nIngesting single file...")
    single_doc = await pipeline.ingest_file(
        "path/to/document.pdf",
        metadata={"priority": "high", "department": "research"}
    )
    print(f"Ingested: {single_doc[0].metadata.get('filename', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(main())