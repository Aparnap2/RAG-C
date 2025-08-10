"""Basic usage example for RAG-C."""

import asyncio
from uni_rag import RAGPipeline, RAGConfig, RAGQuery, Document

async def main():
    config = RAGConfig(
        vector_store={
            "provider": "qdrant",
            "url": "http://localhost:6333",
            "collection_name": "documents",
            "vector_size": 768
        },
        embedding={
            "provider": "google_genai",
            "api_key": "your-google-api-key"
        },
        llm={
            "provider": "google_genai",
            "model": "gemini-1.5-pro",
            "google_api_key": "your-google-api-key"
        },
        ingestion={},
        retrieval={"rrf_k": 60}
    )
    
    pipeline = RAGPipeline(config)
    
    documents = [
        Document(
            id="doc-1",
            content="Artificial intelligence is transforming industries worldwide.",
            metadata={"title": "AI Overview"},
            tenant_id="demo",
            source_tool="manual",
            source_id="demo-1",
            ts_source="2024-01-01T00:00:00Z",
            ts_ingested="2024-01-01T00:01:00Z",
            acl=["tenant:demo"]
        )
    ]
    
    await pipeline.vector_store.add_documents(documents)
    
    query = RAGQuery(query="What is artificial intelligence?", max_results=5)
    response = await pipeline.query(query)
    
    print(f"Answer: {response.answer}")
    print(f"Sources: {len(response.sources)}")

if __name__ == "__main__":
    asyncio.run(main())