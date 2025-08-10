"""Unit tests for vector store implementations."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from RAG-C.vector_stores.base import VectorStoreBase
from RAG-C.vector_stores.qdrant_store import QdrantVectorStore
from RAG-C.vector_stores.astra_store import AstraVectorStore


class TestVectorStoreBase:
    """Test the base vector store interface."""
    
    def test_abstract_methods(self):
        """Test that base class cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorStoreBase()


class TestQdrantVectorStore:
    """Test Qdrant vector store implementation."""
    
    @patch('RAG-C.vector_stores.qdrant_store.QdrantClient')
    @patch('RAG-C.vector_stores.qdrant_store.SentenceTransformer')
    def test_init(self, mock_transformer, mock_client):
        """Test Qdrant store initialization."""
        config = {
            "url": "http://localhost:6333",
            "collection_name": "test_docs",
            "vector_size": 768
        }
        
        # Mock the client and transformer
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value.collections = []
        
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        store = QdrantVectorStore(config)
        
        assert store.collection_name == "test_docs"
        assert store.vector_size == 768
        mock_client.assert_called_once()
        mock_transformer.assert_called_once()
    
    @patch('RAG-C.vector_stores.qdrant_store.QdrantClient')
    @patch('RAG-C.vector_stores.qdrant_store.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_transformer, mock_client, sample_documents):
        """Test adding documents to Qdrant."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value.collections = []
        
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.encode.return_value.tolist.return_value = [[0.1] * 768] * len(sample_documents)
        
        config = {"collection_name": "test_docs"}
        store = QdrantVectorStore(config)
        
        # Test adding documents
        await store.add_documents(sample_documents)
        
        # Verify embeddings were generated
        mock_transformer_instance.encode.assert_called_once()
        
        # Verify upsert was called
        mock_client_instance.upsert.assert_called_once()
        
        # Check the call arguments
        call_args = mock_client_instance.upsert.call_args
        assert call_args[1]["collection_name"] == "test_docs"
        assert len(call_args[1]["points"]) == len(sample_documents)
    
    @patch('RAG-C.vector_stores.qdrant_store.QdrantClient')
    @patch('RAG-C.vector_stores.qdrant_store.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_search(self, mock_transformer, mock_client, sample_documents):
        """Test searching documents in Qdrant."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value.collections = []
        
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.encode.return_value = [[0.1] * 768]
        
        # Mock search results
        mock_result = Mock()
        mock_result.id = "doc-1"
        mock_result.payload = {
            "content": "Test content",
            "metadata": {"title": "Test"},
            "tenant_id": "test-tenant",
            "source_tool": "test-tool",
            "source_id": "source-1",
            "ts_source": "2024-01-01T00:00:00Z",
            "ts_ingested": "2024-01-01T00:01:00Z",
            "acl": ["tenant:test-tenant"]
        }
        mock_client_instance.search.return_value = [mock_result]
        
        config = {"collection_name": "test_docs"}
        store = QdrantVectorStore(config)
        
        # Test search
        results = await store.search("test query", k=5)
        
        # Verify search was called
        mock_client_instance.search.assert_called_once()
        
        # Verify results
        assert len(results) == 1
        assert results[0].id == "doc-1"
        assert results[0].content == "Test content"


class TestAstraVectorStore:
    """Test AstraDB vector store implementation."""
    
    @patch('RAG-C.vector_stores.astra_store.DataAPIClient')
    @patch('RAG-C.vector_stores.astra_store.SentenceTransformer')
    def test_init(self, mock_transformer, mock_client):
        """Test AstraDB store initialization."""
        config = {
            "application_token": "test-token",
            "api_endpoint": "test-endpoint",
            "collection_name": "test_docs",
            "vector_dimension": 768
        }
        
        # Mock the client chain
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_database = Mock()
        mock_client_instance.get_database.return_value = mock_database
        
        mock_collection = Mock()
        mock_database.get_collection.return_value = mock_collection
        
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        
        store = AstraVectorStore(config)
        
        assert store.collection_name == "test_docs"
        assert store.vector_dimension == 768
        mock_client.assert_called_once_with("test-token")
        mock_transformer.assert_called_once()
    
    @patch('RAG-C.vector_stores.astra_store.DataAPIClient')
    @patch('RAG-C.vector_stores.astra_store.SentenceTransformer')
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_transformer, mock_client, sample_documents):
        """Test adding documents to AstraDB."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_database = Mock()
        mock_client_instance.get_database.return_value = mock_database
        
        mock_collection = Mock()
        mock_database.get_collection.return_value = mock_collection
        
        mock_transformer_instance = Mock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.encode.return_value.tolist.return_value = [[0.1] * 768] * len(sample_documents)
        
        config = {
            "application_token": "test-token",
            "api_endpoint": "test-endpoint",
            "collection_name": "test_docs"
        }
        store = AstraVectorStore(config)
        
        # Test adding documents
        await store.add_documents(sample_documents)
        
        # Verify embeddings were generated
        mock_transformer_instance.encode.assert_called_once()
        
        # Verify insert_many was called
        mock_collection.insert_many.assert_called_once()
        
        # Check the call arguments
        call_args = mock_collection.insert_many.call_args[0][0]
        assert len(call_args) == len(sample_documents)
        assert all("$vector" in doc for doc in call_args)