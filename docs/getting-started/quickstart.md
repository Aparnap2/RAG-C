# Quick Start

## Basic RAG Pipeline

```python
from RAG-C import RAGPipeline, RAGConfig, RAGQuery, Document

# Configure pipeline
config = RAGConfig(
    vector_store={
        "provider": "qdrant",
        "url": "http://localhost:6333",
        "collection_name": "documents"
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

# Create pipeline
pipeline = RAGPipeline(config)

# Add documents
documents = [
    Document(
        id="doc-1",
        content="AI is transforming industries worldwide.",
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

# Query
query = RAGQuery(query="What is AI?", max_results=5)
response = await pipeline.query(query)

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
```

## Hybrid Retrieval

```python
from RAG-C.retrieval_hybrid import HybridRetriever

# Add text index for hybrid search
config.text_index = {
    "provider": "opensearch",
    "host": "localhost",
    "port": 9200
}

pipeline = RAGPipeline(config)

# Hybrid retrieval automatically enabled
response = await pipeline.query(query)
```

## Factory Functions

```python
from RAG-C import get_vector_store, get_embedding_client

# Direct component usage
vector_store = get_vector_store({
    "provider": "qdrant",
    "url": "http://localhost:6333"
})

embedding_client = get_embedding_client({
    "provider": "google_genai",
    "api_key": "your-key"
})
```