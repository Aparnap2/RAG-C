"""Unit tests for RAG pipeline."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from RAG-C.pipeline import RAGPipeline
from RAG-C.models import RAGQuery, RAGResponse


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @patch('RAG-C.pipeline.get_vector_store')
    @patch('RAG-C.pipeline.get_text_index')
    @patch('RAG-C.pipeline.get_llm')
    def test_init(self, mock_get_llm, mock_get_text_index, mock_get_vector_store, test_config):
        """Test pipeline initialization."""
        # Setup mocks
        mock_vector_store = AsyncMock()
        mock_text_index = AsyncMock()
        mock_llm = Mock()
        
        mock_get_vector_store.return_value = mock_vector_store
        mock_get_text_index.return_value = mock_text_index
        mock_get_llm.return_value = mock_llm
        
        # Create pipeline
        pipeline = RAGPipeline(test_config)
        
        # Verify initialization
        assert pipeline.vector_store == mock_vector_store
        assert pipeline.text_index == mock_text_index
        assert pipeline.llm == mock_llm
        assert pipeline.hybrid_retriever is not None
        
        # Verify factory functions were called
        mock_get_vector_store.assert_called_once_with(test_config.vector_store)
        mock_get_text_index.assert_called_once_with(test_config.text_index)
        mock_get_llm.assert_called_once_with(test_config.llm)
    
    @patch('RAG-C.pipeline.get_vector_store')
    @patch('RAG-C.pipeline.get_text_index')
    @patch('RAG-C.pipeline.get_llm')
    @pytest.mark.asyncio
    async def test_query_hybrid_retrieval(self, mock_get_llm, mock_get_text_index, 
                                        mock_get_vector_store, test_config, sample_documents):
        """Test query with hybrid retrieval."""
        # Setup mocks
        mock_vector_store = AsyncMock()
        mock_text_index = AsyncMock()
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Test response about AI"
        
        mock_get_vector_store.return_value = mock_vector_store
        mock_get_text_index.return_value = mock_text_index
        mock_get_llm.return_value = mock_llm
        
        # Mock hybrid retriever
        with patch('RAG-C.pipeline.HybridRetriever') as mock_hybrid_class:
            mock_hybrid_retriever = AsyncMock()
            mock_hybrid_class.return_value = mock_hybrid_retriever
            
            # Mock retrieval results
            mock_hybrid_retriever.retrieve.return_value = [
                {
                    "id": "doc-1",
                    "content": "AI is transforming industries",
                    "metadata": {"title": "AI Overview"},
                    "tenant_id": "test-tenant",
                    "source_tool": "test-tool",
                    "source_id": "source-1",
                    "ts_source": "2024-01-01T00:00:00Z",
                    "ts_ingested": "2024-01-01T00:01:00Z",
                    "acl": ["tenant:test-tenant"],
                    "score": 0.9
                }
            ]
            
            # Create pipeline
            pipeline = RAGPipeline(test_config)
            
            # Test query
            query = RAGQuery(
                query="What is artificial intelligence?",
                max_results=5,
                source_type="web"
            )
            
            response = await pipeline.query(query)
            
            # Verify response
            assert isinstance(response, RAGResponse)
            assert response.answer == "Test response about AI"
            assert len(response.sources) == 1
            assert response.sources[0].content == "AI is transforming industries"
            
            # Verify hybrid retriever was called
            mock_hybrid_retriever.retrieve.assert_called_once()
            call_args = mock_hybrid_retriever.retrieve.call_args
            assert call_args[0][0] == "What is artificial intelligence?"  # query
            assert "tenant_id" in call_args[0][1]  # filters
            assert call_args[1]["top_k"] == 5  # top_k
    
    @patch('RAG-C.pipeline.get_vector_store')
    @patch('RAG-C.pipeline.get_text_index')
    @patch('RAG-C.pipeline.get_llm')
    @pytest.mark.asyncio
    async def test_query_vector_only_fallback(self, mock_get_llm, mock_get_text_index, 
                                            mock_get_vector_store, test_config, sample_documents):
        """Test query fallback to vector-only when text index is None."""
        # Setup mocks
        mock_vector_store = AsyncMock()
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Vector-only response"
        
        mock_get_vector_store.return_value = mock_vector_store
        mock_get_text_index.return_value = None  # No text index
        mock_get_llm.return_value = mock_llm
        
        # Mock vector store search
        mock_vector_store.search.return_value = sample_documents[:1]
        
        # Create config without text index
        config_no_text = test_config.model_copy()
        config_no_text.text_index = None
        
        # Create pipeline
        pipeline = RAGPipeline(config_no_text)
        
        # Test query
        query = RAGQuery(
            query="What is AI?",
            max_results=3,
            source_type="web"
        )
        
        response = await pipeline.query(query)
        
        # Verify response
        assert isinstance(response, RAGResponse)
        assert response.answer == "Vector-only response"
        assert len(response.sources) == 1
        
        # Verify vector store was called directly
        mock_vector_store.search.assert_called_once_with("What is AI?", 3)
        
        # Verify no hybrid retriever was created
        assert pipeline.hybrid_retriever is None
    
    @patch('RAG-C.pipeline.get_vector_store')
    @patch('RAG-C.pipeline.get_text_index')
    @patch('RAG-C.pipeline.get_llm')
    @patch('RAG-C.pipeline.get_memory')
    @pytest.mark.asyncio
    async def test_query_with_memory(self, mock_get_memory, mock_get_llm, mock_get_text_index, 
                                   mock_get_vector_store, test_config):
        """Test query with conversation memory."""
        # Setup mocks
        mock_vector_store = AsyncMock()
        mock_text_index = AsyncMock()
        mock_llm = Mock()
        mock_memory = AsyncMock()
        
        mock_llm.invoke.return_value.content = "Response with memory context"
        mock_memory.get_context.return_value = "Previous conversation context"
        
        mock_get_vector_store.return_value = mock_vector_store
        mock_get_text_index.return_value = mock_text_index
        mock_get_llm.return_value = mock_llm
        mock_get_memory.return_value = mock_memory
        
        # Mock hybrid retriever
        with patch('RAG-C.pipeline.HybridRetriever') as mock_hybrid_class:
            mock_hybrid_retriever = AsyncMock()
            mock_hybrid_class.return_value = mock_hybrid_retriever
            mock_hybrid_retriever.retrieve.return_value = []
            
            # Create pipeline
            pipeline = RAGPipeline(test_config)
            
            # Test query
            query = RAGQuery(
                query="Follow up question",
                max_results=5,
                source_type="web"
            )
            
            response = await pipeline.query(query)
            
            # Verify memory was used
            mock_memory.get_context.assert_called_once_with("Follow up question")
            mock_memory.add_context.assert_called_once_with(query, "Response with memory context")
            
            # Verify response includes memory context
            assert response.answer == "Response with memory context"
    
    @patch('RAG-C.pipeline.get_vector_store')
    @patch('RAG-C.pipeline.get_text_index')
    @patch('RAG-C.pipeline.get_llm')
    @patch('RAG-C.pipeline.get_knowledge_graph')
    @pytest.mark.asyncio
    async def test_query_with_knowledge_graph(self, mock_get_kg, mock_get_llm, mock_get_text_index, 
                                            mock_get_vector_store, test_config):
        """Test query with knowledge graph."""
        # Setup mocks
        mock_vector_store = AsyncMock()
        mock_text_index = AsyncMock()
        mock_llm = Mock()
        mock_kg = AsyncMock()
        
        mock_llm.invoke.return_value.content = "Response with graph context"
        mock_kg.query_relations.return_value = [
            {"properties": {"content": "Graph relation 1"}},
            {"properties": {"content": "Graph relation 2"}}
        ]
        
        mock_get_vector_store.return_value = mock_vector_store
        mock_get_text_index.return_value = mock_text_index
        mock_get_llm.return_value = mock_llm
        mock_get_kg.return_value = mock_kg
        
        # Mock hybrid retriever
        with patch('RAG-C.pipeline.HybridRetriever') as mock_hybrid_class:
            mock_hybrid_retriever = AsyncMock()
            mock_hybrid_class.return_value = mock_hybrid_retriever
            mock_hybrid_retriever.retrieve.return_value = []
            
            # Create pipeline
            pipeline = RAGPipeline(test_config)
            
            # Test query
            query = RAGQuery(
                query="Graph query",
                max_results=5,
                source_type="web"
            )
            
            response = await pipeline.query(query)
            
            # Verify knowledge graph was queried
            mock_kg.query_relations.assert_called_once_with("Graph query")
            
            # Verify response
            assert response.answer == "Response with graph context"
            assert "Graph relation 1" in response.context
            assert "Graph relation 2" in response.context