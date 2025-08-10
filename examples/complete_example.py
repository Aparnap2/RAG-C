"""Complete example showcasing all UNI-RAG features."""

import asyncio
from uni_rag import RAGPipeline, RAGConfig, RAGQuery

async def main():
    # Complete configuration
    config = RAGConfig(
        # MCP configuration
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
        
        # Storage configuration
        vector_store={
            "provider": "qdrant",
            "url": "http://localhost:6333",
            "collection_name": "uni_rag_docs"
        },
        text_index={
            "provider": "opensearch",
            "host": "localhost",
            "port": 9200
        },
        
        # Embeddings
        embedding={
            "provider": "google_genai",
            "api_key": "your-api-key"
        },
        
        # LLM
        llm={
            "provider": "google_genai",
            "model": "gemini-1.5-pro",
            "google_api_key": "your-api-key"
        },
        
        # Advanced processing
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
            "quality_threshold": 0.75
        },
        
        # Enhanced reranking
        reranker={
            "threshold": 0.8,
            "recency_weight": 0.1,
            "entity_weight": 0.2
        },
        
        # Ingestion configuration
        ingestion={
            "file": {
                "use_docling": True,
                "max_concurrent": 5
            },
            "web": {
                "use_crawl4ai": True,
                "max_concurrent": 3,
                "timeout": 30
            }
        }
    )
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    print("🚀 UNI-RAG Complete Example")
    print("=" * 50)
    
    # 1. Ingest mixed sources
    print("\n📥 Ingesting mixed sources...")
    sources = {
        "files": ["documents/report.pdf", "documents/manual.docx"],
        "urls": ["https://example.com/article", "https://example.com/blog"],
        "directories": ["documents/research/"]
    }
    
    try:
        documents = await pipeline.ingest_mixed_sources(
            sources,
            metadata={"project": "demo", "priority": "high"}
        )
        print(f"✅ Ingested {len(documents)} documents")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        # Continue with example query
    
    # 2. Single source ingestion examples
    print("\n📄 Single source examples...")
    
    # File ingestion
    try:
        docs = await pipeline.ingest("example.pdf", "file")
        print(f"✅ File: {len(docs)} documents")
    except Exception as e:
        print(f"⚠️  File ingestion: {e}")
    
    # URL ingestion
    try:
        docs = await pipeline.ingest("https://example.com", "url")
        print(f"✅ URL: {len(docs)} documents")
    except Exception as e:
        print(f"⚠️  URL ingestion: {e}")
    
    # Auto-detection
    try:
        docs = await pipeline.ingest("documents/", "auto")  # Auto-detects directory
        print(f"✅ Auto-detect: {len(docs)} documents")
    except Exception as e:
        print(f"⚠️  Auto-detect: {e}")
    
    # 3. Advanced querying
    print("\n🔍 Advanced querying with LangGraph...")
    
    queries = [
        "What are the key benefits of artificial intelligence?",
        "How does machine learning improve business processes?",
        "What are the latest trends in technology?"
    ]
    
    for i, query_text in enumerate(queries, 1):
        print(f"\n🤔 Query {i}: {query_text}")
        
        try:
            query = RAGQuery(
                query=query_text,
                max_results=5,
                source_type="mixed"
            )
            
            response = await pipeline.query(query)
            
            print(f"💡 Answer: {response.answer[:200]}...")
            print(f"📊 Context length: {len(response.context)} chars")
            print(f"📚 Sources: {len(response.sources)}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # 4. Demonstrate different ingestion methods
    print("\n🔧 Advanced ingestion examples...")
    
    # Sitemap ingestion
    try:
        docs = await pipeline.ingest("https://example.com/sitemap.xml", "sitemap")
        print(f"✅ Sitemap: {len(docs)} documents")
    except Exception as e:
        print(f"⚠️  Sitemap: {e}")
    
    print("\n🎉 Complete example finished!")
    print("=" * 50)
    print("Features demonstrated:")
    print("✓ Multi-chunk strategies (200/400/800 tokens)")
    print("✓ LangGraph orchestration with 6-stage workflow")
    print("✓ Docling document processing")
    print("✓ Crawl4AI web scraping")
    print("✓ Unified ingestion (files, URLs, directories)")
    print("✓ Hybrid retrieval with RRF")
    print("✓ Cross-encoder reranking")
    print("✓ MCP-first architecture")

if __name__ == "__main__":
    asyncio.run(main())