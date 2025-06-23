import pytest
from rag.pipeline import RAGPipeline
from rag.models import RAGQuery

@pytest.mark.asyncio
async def test_rag_query():
    pipeline = RAGPipeline(vector_provider="qdrant")
    query = RAGQuery(query="What is AI?", source_type="web", source_url="https://example.com")
    response = await pipeline.query(query)
    assert response.answer
    assert len(response.sources) <= 5 