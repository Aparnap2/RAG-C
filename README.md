# Universal RAG Stack

A modular, production-ready Retrieval-Augmented Generation (RAG) stack for AI-driven applications. Integrates LangChain, LlamaIndex, Docling, PyPDF, AstraDB/Qdrant, Graphiti/Neo4j, Mem0, and Crawl4AI. Exposes a FastAPI backend for easy integration with Next.js/React frontends.

## Features
- Ingest PDFs (Docling/PyPDF) and web pages (Crawl4AI)
- Store embeddings in AstraDB or Qdrant
- Maintain knowledge graphs in Neo4j (Graphiti)
- Cache conversational context with Mem0
- RAG pipeline with LangChain and LlamaIndex (Google Gemini)
- Modular, type-safe (Pydantic), and reusable

## Project Structure
```
rag-stack/
├── rag/
│   ├── ingestion.py        # Document ingestion (PDFs, web)
│   ├── vector_store.py     # AstraDB/Qdrant vector storage
│   ├── knowledge_graph.py  # Graphiti/Neo4j integration
│   ├── memory.py           # Mem0 for conversational memory
│   ├── pipeline.py         # LangChain/LlamaIndex RAG pipeline
│   └── models.py           # Pydantic models
├── api/
│   └── main.py             # FastAPI endpoints
├── .env                    # Environment variables
├── requirements.txt        # Dependencies
└── README.md
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `.env` with your API keys and endpoints.
3. Run Neo4j (locally or via Docker/Aura).
4. Start FastAPI:
   ```bash
   uvicorn api.main:app --reload
   ```

## Usage
- Ingest PDFs or web pages and query with context-aware RAG pipeline.
- Integrate with your Next.js frontend via provided API endpoints.

## Testing
- Run unit tests with `pytest tests/`.

## Extending
- Swap vector store, LLM, or add new ingestion formats easily.

---
For more details, see code comments and docstrings. 