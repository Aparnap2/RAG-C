"""Test configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from RAG-C.models import Document, RAGConfig


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return Document(
        id="test-doc-1",
        content="This is a test document about artificial intelligence and machine learning.",
        metadata={"title": "Test Document", "author": "Test Author"},
        tenant_id="test-tenant",
        source_tool="test-tool",
        source_id="test-source-1",
        ts_source="2024-01-01T00:00:00Z",
        ts_ingested="2024-01-01T00:01:00Z",
        acl=["tenant:test-tenant"]
    )


@pytest.fixture
def sample_documents():
    """Multiple sample documents for testing."""
    return [
        Document(
            id="doc-1",
            content="Artificial intelligence is transforming industries.",
            metadata={"title": "AI Overview"},
            tenant_id="test-tenant",
            source_tool="test-tool",
            source_id="source-1",
            ts_source="2024-01-01T00:00:00Z",
            ts_ingested="2024-01-01T00:01:00Z",
            acl=["tenant:test-tenant"]
        ),
        Document(
            id="doc-2", 
            content="Machine learning algorithms require large datasets.",
            metadata={"title": "ML Basics"},
            tenant_id="test-tenant",
            source_tool="test-tool",
            source_id="source-2",
            ts_source="2024-01-01T01:00:00Z",
            ts_ingested="2024-01-01T01:01:00Z",
            acl=["tenant:test-tenant"]
        ),
        Document(
            id="doc-3",
            content="Deep learning uses neural networks with multiple layers.",
            metadata={"title": "Deep Learning"},
            tenant_id="test-tenant",
            source_tool="test-tool",
            source_id="source-3",
            ts_source="2024-01-01T02:00:00Z",
            ts_ingested="2024-01-01T02:01:00Z",
            acl=["tenant:test-tenant"]
        )
    ]


@pytest.fixture
def test_config():
    """Test RAG configuration."""
    return RAGConfig(
        mcp={
            "servers": {},
            "tenants": {
                "test-tenant": {
                    "allowed_tools": ["test-tool"]
                }
            }
        },
        ingestion={
            "pdf": {},
            "web": {}
        },
        vector_store={
            "provider": "qdrant",
            "url": "http://localhost:6333",
            "collection_name": "test_documents",
            "vector_size": 768,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        text_index={
            "provider": "opensearch",
            "host": "localhost",
            "port": 9200,
            "index_name": "test_documents"
        },
        knowledge_graph={
            "provider": "neo4j"
        },
        retrieval={
            "rrf_k": 60,
            "vector_weight": 1.0,
            "bm25_weight": 1.0
        },
        reranker={
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        llm={
            "provider": "google_genai",
            "model": "gemini-1.5-pro"
        }
    )


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = AsyncMock()
    mock.add_documents = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.get_documents = AsyncMock(return_value=[])
    mock.delete_documents = AsyncMock()
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_text_index():
    """Mock text index for testing."""
    mock = AsyncMock()
    mock.add_documents = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.delete_documents = AsyncMock()
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="Test response"))
    return mock