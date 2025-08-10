"""Embedding implementations."""

from abc import ABC, abstractmethod
from typing import List
import google.generativeai as genai


class EmbeddingBase(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str], model: str) -> List[List[float]]:
        pass


class GoogleGenAIEmbedding(EmbeddingBase):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
    async def embed_documents(self, texts: List[str], model: str = "models/embedding-001") -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(model=model, content=text)
            embeddings.append(result['embedding'])
        return embeddings