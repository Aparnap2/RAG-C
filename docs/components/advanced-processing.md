# Advanced Processing

UNI-RAG implements Google Cloud's Advanced RAG techniques for superior context quality.

## Multi-Chunk Strategies

Store the same content in multiple chunk sizes for optimal retrieval:

```python
config = RAGConfig(
    chunking={
        "chunk_sizes": [200, 400, 800],  # Multiple sizes
        "overlap_ratio": 0.15            # 15% overlap
    }
)
```

**Benefits:**
- Small chunks: Precise information retrieval
- Medium chunks: Balanced context and precision  
- Large chunks: Comprehensive context

## LangGraph Orchestration

Multi-LLM workflow using LangGraph:

```python
config = RAGConfig(
    llm_orchestration={
        "llm_tool": "llm.generate",
        "enable_preprocessing": True,
        "retrieval_k": 50,
        "final_k": 5,
        "quality_threshold": 0.7
    }
)
```

**Workflow Steps:**
1. **Query Preprocessing**: Spelling correction, expansion, synonyms
2. **Hybrid Retrieval**: Vector + BM25 with RRF fusion
3. **Cross-Encoder Reranking**: Quality scoring and filtering
4. **Context Summarization**: Condense retrieved information
5. **Grounded Generation**: Citations and evidence-based responses
6. **Self-Evaluation**: Response quality assessment

## Docling Integration

Advanced document processing with Docling:

```python
# Automatic format detection and processing
documents = await pipeline.ingest_file("document.pdf")

# Batch processing
documents = await pipeline.ingest_directory(
    "documents/",
    file_patterns=["*.pdf", "*.docx", "*.pptx"]
)
```

**Supported Formats:**
- PDF documents
- Microsoft Word (.docx)
- PowerPoint (.pptx)
- Excel (.xlsx)
- Text files (.txt, .md)

## Hypothetical Questions

Generate questions each chunk could answer:

```python
# Automatically generated during ingestion
chunk.hypothetical_questions = [
    "What are the benefits of AI?",
    "How does machine learning work?",
    "What are AI applications?"
]
```

## Enhanced Metadata

LLM-powered metadata enrichment:

```python
chunk.metadata_tags = {
    "topic": "artificial_intelligence",
    "category": "technology",
    "entities": ["AI", "machine learning", "neural networks"],
    "sentiment": "positive"
}
```

## Quality Scoring

Authority and quality metrics:

```python
chunk.quality_score = 0.85  # 0.0 to 1.0

# Factors:
# - Source authority
# - Content length appropriateness  
# - Metadata richness
# - Hypothetical questions count
```

## Configuration Example

Complete advanced configuration:

```python
config = RAGConfig(
    # MCP for LLM calls
    mcp={
        "servers": {"llm_server": {"transport": "stdio"}},
        "tenants": {"demo": {"allowed_tools": ["llm.generate"]}}
    },
    
    # Multi-chunk processing
    chunking={
        "chunk_sizes": [200, 400, 800],
        "overlap_ratio": 0.15
    },
    
    # LangGraph orchestration
    llm_orchestration={
        "enable_preprocessing": True,
        "quality_threshold": 0.7
    },
    
    # Enhanced reranking
    reranker={
        "threshold": 0.8,
        "recency_weight": 0.1
    }
)
```