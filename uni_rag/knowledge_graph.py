from rag.interfaces import KnowledgeGraphBase
from graphiti_core import Graphiti
from graphiti_core.nodes import Node
from graphiti_core.edges import Edge
from rag.models import Document
import os
from uuid import uuid4
from typing import List

class KnowledgeGraph(KnowledgeGraphBase):
    def __init__(self):
        self.graph = Graphiti(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username=os.getenv("NEO4J_USERNAME"),
            neo4j_password=os.getenv("NEO4J_PASSWORD")
        )

    async def add_document(self, document: Document):
        # Create document node
        doc_node = Node(
            id=document.id,
            type="Document",
            properties={"content": document.content, "source": document.metadata.get("source")}
        )
        self.graph.add_node(doc_node)

        # Extract entities (simplified example)
        entities = ["entity1", "entity2"]  # Replace with NLP-based entity extraction
        for entity in entities:
            entity_node = Node(id=str(uuid4()), type="Entity", properties={"name": entity})
            self.graph.add_node(entity_node)
            edge = Edge(source_id=document.id, target_id=entity_node.id, type="CONTAINS")
            self.graph.add_edge(edge)

    async def query_relations(self, query: str) -> List[dict]:
        # Simplified query for related entities
        results = self.graph.query_nodes(type="Document", properties={"content": query})
        return [{"id": node.id, "properties": node.properties} for node in results] 