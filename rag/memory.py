from rag.interfaces import MemoryBase
from mem0 import Memory
from rag.models import RAGQuery
import os

class ConversationMemory(MemoryBase):
    def __init__(self):
        self.memory = Memory(api_key=os.getenv("MEM0_API_KEY"))

    async def add_context(self, query: RAGQuery, response: str):
        self.memory.add(
            data={"query": query.query, "response": response},
            user_id="system",
            metadata={"source_type": query.source_type}
        )

    async def get_context(self, query: str) -> str:
        memories = self.memory.search(query=query, user_id="system")
        return "\n".join([mem["data"]["response"] for mem in memories]) if memories else "" 