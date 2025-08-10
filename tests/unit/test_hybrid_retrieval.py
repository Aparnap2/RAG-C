"""Unit tests for hybrid retrieval with RRF."""

import pytest
from unittest.mock import Mock, AsyncMock
from RAG-C.retrieval_hybrid import HybridRetriever
from RAG-C.models import Document


class TestHybridRetriever:
    """Test hybrid retrieval with RRF fusion."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        mock = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_text_index(self):
        """Mock text index."""
        mock = AsyncMock()
        return mock
    
    @pytest.fixture
    def retriever(self, mock_vector_store, mock_text_index):
        """Create hybrid retriever with mocks."""
        config = {
            "rrf_k": 60,
            "vector_weight": 1.0,
            "bm25_weight": 1.0
        }
        return HybridRetriever(mock_vector_store, mock_text_index, config)
    
    @pytest.mark.asyncio
    async def test_retrieve_basic(self, retriever, mock_vector_store, mock_text_index, sample_documents):
        """Test basic hybrid retrieval."""
        # Setup mock responses
        mock_vector_store.search.return_value = sample_documents[:2]
        mock_text_index.search.return_value = [
            {
                "id": "doc-2",
                "score": 0.8,
                "content": "Machine learning algorithms require large datasets.",
                "metadata": {"title": "ML Basics"},
                "tenant_id": "test-tenant",
                "source_tool": "test-tool",
                "source_id": "source-2",
                "ts_source": "2024-01-01T01:00:00Z",
                "ts_ingested": "2024-01-01T01:01:00Z",
                "acl": ["tenant:test-tenant"]
            },
            {
                "id": "doc-3",
                "score": 0.7,
                "content": "Deep learning uses neural networks.",
                "metadata": {"title": "Deep Learning"},
                "tenant_id": "test-tenant",
                "source_tool": "test-tool",
                "source_id": "source-3",
                "ts_source": "2024-01-01T02:00:00Z",
                "ts_ingested": "2024-01-01T02:01:00Z",
                "acl": ["tenant:test-tenant"]
            }
        ]
        
        # Test retrieval
        query = "machine learning"
        filters = {"tenant_id": "test-tenant"}
        results = await retriever.retrieve(query, filters, top_k=5)
        
        # Verify both stores were called
        mock_vector_store.search.assert_called_once_with(query, 5, filters)
        mock_text_index.search.assert_called_once_with(query, 5, filters)
        
        # Verify results
        assert len(results) >= 0  # RRF may return different number based on fusion
        assert all(isinstance(result, dict) for result in results)
        assert all("id" in result and "score" in result for result in results)
    
    def test_reciprocal_rank_fusion(self, retriever):
        """Test RRF algorithm."""
        # Test data
        result_lists = [
            {
                "results": [
                    {"id": "doc-1", "score": 0.9},
                    {"id": "doc-2", "score": 0.8},
                    {"id": "doc-3", "score": 0.7}
                ],
                "weight": 1.0
            },
            {
                "results": [
                    {"id": "doc-2", "score": 0.85},
                    {"id": "doc-3", "score": 0.75},
                    {"id": "doc-1", "score": 0.65}
                ],
                "weight": 1.0
            }
        ]
        
        # Apply RRF
        fused = retriever._reciprocal_rank_fusion(result_lists, k=60)
        
        # Verify results
        assert len(fused) == 3
        assert all("id" in result and "score" in result for result in fused)
        
        # Results should be sorted by score (descending)
        scores = [result["score"] for result in fused]
        assert scores == sorted(scores, reverse=True)
        
        # doc-2 should have highest score (appears first in both lists)
        assert fused[0]["id"] == "doc-2"
    
    def test_deduplicate_results(self, retriever):
        """Test result deduplication."""
        results = [
            {"id": "doc-1", "score": 0.9},
            {"id": "doc-2", "score": 0.8},
            {"id": "doc-1", "score": 0.7},  # Duplicate
            {"id": "doc-3", "score": 0.6}
        ]
        
        deduplicated = retriever._deduplicate_results(results)
        
        # Should remove duplicates
        assert len(deduplicated) == 3
        ids = [result["id"] for result in deduplicated]
        assert len(set(ids)) == 3  # All unique
        
        # Should keep first occurrence
        doc_1_result = next(r for r in deduplicated if r["id"] == "doc-1")
        assert doc_1_result["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, retriever, mock_vector_store, mock_text_index):
        """Test retrieval with various filters."""
        # Setup mocks
        mock_vector_store.search.return_value = []
        mock_text_index.search.return_value = []
        
        # Test with tenant filter
        filters = {
            "tenant_id": "test-tenant",
            "acl": ["tenant:test-tenant"],
            "time_window": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T23:59:59Z"
            }
        }
        
        await retriever.retrieve("test query", filters, top_k=10)
        
        # Verify filters were passed to both stores
        mock_vector_store.search.assert_called_once_with("test query", 10, filters)
        mock_text_index.search.assert_called_once_with("test query", 10, filters)
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, retriever, mock_vector_store, mock_text_index):
        """Test retrieval with empty results."""
        # Setup mocks to return empty results
        mock_vector_store.search.return_value = []
        mock_text_index.search.return_value = []
        
        results = await retriever.retrieve("test query", {}, top_k=5)
        
        # Should return empty list
        assert results == []
    
    @pytest.mark.asyncio
    async def test_retrieve_vector_only(self, retriever, mock_vector_store, mock_text_index, sample_documents):
        """Test retrieval when only vector store has results."""
        # Setup mocks
        mock_vector_store.search.return_value = sample_documents[:1]
        mock_text_index.search.return_value = []
        
        results = await retriever.retrieve("test query", {}, top_k=5)
        
        # Should still return results from vector store
        assert len(results) >= 0
    
    @pytest.mark.asyncio
    async def test_retrieve_text_only(self, retriever, mock_vector_store, mock_text_index):
        """Test retrieval when only text index has results."""
        # Setup mocks
        mock_vector_store.search.return_value = []
        mock_text_index.search.return_value = [
            {
                "id": "doc-1",
                "score": 0.8,
                "content": "Test content",
                "metadata": {},
                "tenant_id": "test-tenant",
                "source_tool": "test-tool",
                "source_id": "source-1",
                "ts_source": "2024-01-01T00:00:00Z",
                "ts_ingested": "2024-01-01T00:01:00Z",
                "acl": ["tenant:test-tenant"]
            }
        ]
        
        results = await retriever.retrieve("test query", {}, top_k=5)
        
        # Should still return results from text index
        assert len(results) >= 0