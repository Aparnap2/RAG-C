from rag.interfaces import VectorStoreBase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import AstraDB, Qdrant
from rag.models import Document
import os
from uuid import uuid4
from typing import List

class VectorStore(VectorStoreBase):
    def __init__(self, provider: str = "astradb"):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.provider = provider
        if provider == "astradb":
            self.store = AstraDB(
                collection_name="rag_collection",
                embedding=self.embedding_model,
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
            )
        elif provider == "qdrant":
            self.store = Qdrant.from_texts(
                texts=[],
                embedding=self.embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="rag_collection"
            )

    async def add_documents(self, documents: List[Document]):
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if self.provider == "astradb":
            self.store.add_texts(texts, metadatas)
        elif self.provider == "qdrant":
            self.store.add_texts(texts, metadatas)

    async def search(self, query: str, k: int = 5) -> List[Document]:
        results = self.store.similarity_search(query, k=k)
        return [Document(id=str(uuid4()), content=doc.page_content, metadata=doc.metadata) for doc in results] 