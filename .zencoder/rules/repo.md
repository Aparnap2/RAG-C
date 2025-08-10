---
description: Repository Information Overview
alwaysApply: true
---

# Universal RAG Stack Information

## Summary
A modular, production-ready Retrieval-Augmented Generation (RAG) stack for AI-driven applications. Integrates LangChain, LlamaIndex, and various tools for document ingestion, vector storage, knowledge graphs, and conversational memory. Exposes a FastAPI backend for easy integration with frontend applications.

## Structure
- **rag/**: Core RAG components (ingestion, vector store, knowledge graph, memory, pipeline)
- **api/**: FastAPI endpoints for RAG functionality
- **tests/**: Test files for the RAG pipeline

## Language & Runtime
**Language**: Python
**Build System**: pip
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- langchain-google-genai: Google Generative AI integration
- llama-index: RAG framework
- docling/pypdf: PDF document processing
- crawl4ai: Web page ingestion
- astrapy/qdrant-client: Vector database clients
- graphiti-core: Knowledge graph integration
- mem0: Conversational memory
- fastapi/uvicorn: API server
- pydantic: Data validation and settings management

**Development Dependencies**:
- pytest: Testing framework

## Build & Installation
```bash
pip install -r requirements.txt
```

## Main Components
**Entry Point**: api/main.py (FastAPI application)
**Configuration**: Environment variables or JSON configuration file
**Core Module**: rag/pipeline.py (RAGPipeline class)

## Usage
The application provides a FastAPI server with endpoints for:
- Querying the RAG pipeline: POST /api/rag/query
- Ingesting documents: POST /api/rag/ingest

Start the server with:
```bash
uvicorn api.main:app --reload
```

## Testing
**Framework**: pytest
**Test Location**: tests/
**Run Command**:
```bash
pytest tests/
```

## Architecture
The project follows a modular architecture with:
- Factory pattern for component creation (factory.py)
- Pydantic models for type safety (models.py)
- Configuration-driven setup (config.py)
- Pluggable components for vector stores, LLMs, and ingestion methods