"""
Graph sink for storing documents in a temporal knowledge graph.
Implements validity windows, provenance tracking, and conflict resolution.
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class GraphSink:
    """
    Sink for storing documents in a temporal knowledge graph.
    """
    def __init__(self, graph_client, config: Dict[str, Any]):
        self.graph_client = graph_client
        self.config = config
        self.entity_extractors = self._load_entity_extractors()
        self.relation_extractors = self._load_relation_extractors()
        
    def _load_entity_extractors(self) -> List[Any]:
        """Load entity extractors based on config"""
        extractors = []
        
        # In a real implementation, this would load actual extractors
        # For now, just return a dummy extractor
        class DummyEntityExtractor:
            async def extract(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
                # Extract some dummy entities
                return [
                    {
                        "type": "person",
                        "text": "John Doe",
                        "start": 0,
                        "end": 8,
                        "confidence": 0.9
                    }
                ]
                
        extractors.append(DummyEntityExtractor())
        return extractors
        
    def _load_relation_extractors(self) -> List[Any]:
        """Load relation extractors based on config"""
        extractors = []
        
        # In a real implementation, this would load actual extractors
        # For now, just return a dummy extractor
        class DummyRelationExtractor:
            async def extract(self, document: Dict[str, Any], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                # Extract some dummy relations
                if len(entities) >= 2:
                    return [
                        {
                            "type": "works_for",
                            "source": entities[0],
                            "target": entities[1],
                            "confidence": 0.8
                        }
                    ]
                return []
                
        extractors.append(DummyRelationExtractor())
        return extractors
        
    async def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document and store entities and relations in the graph
        
        Args:
            document: Normalized document
            
        Returns:
            Processing result
        """
        # Extract entities
        entities = []
        for extractor in self.entity_extractors:
            entities.extend(await extractor.extract(document))
            
        # Extract relations
        relations = []
        for extractor in self.relation_extractors:
            relations.extend(await extractor.extract(document, entities))
            
        # Store entities and relations
        nodes_created = await self._store_entities(entities, document)
        edges_created = await self._store_relations(relations, document)
        
        return {
            "document_id": document["id"],
            "nodes_created": nodes_created,
            "edges_created": edges_created
        }
        
    async def _store_entities(self, entities: List[Dict[str, Any]], document: Dict[str, Any]) -> int:
        """
        Store entities in the graph
        
        Args:
            entities: Extracted entities
            document: Source document
            
        Returns:
            Number of nodes created
        """
        nodes_created = 0
        
        for entity in entities:
            # Create a unique ID for the entity
            entity_id = f"{document['tenant_id']}:{entity['type']}:{entity['text']}"
            
            # Create node properties
            properties = {
                "type": entity["type"],
                "text": entity["text"],
                "summary": entity["text"],
                "confidence": entity.get("confidence", 1.0),
                "tenant_id": document["tenant_id"],
                "provenance": {
                    "document_id": document["id"],
                    "source_tool": document["source_tool"],
                    "ts_extracted": datetime.now().isoformat()
                }
            }
            
            # Check if node exists
            existing_node = await self.graph_client.get_node(entity_id)
            
            if existing_node:
                # Update existing node
                await self.graph_client.update_node(entity_id, properties)
            else:
                # Create new node
                await self.graph_client.create_node(entity_id, entity["type"], properties)
                nodes_created += 1
                
        return nodes_created
        
    async def _store_relations(self, relations: List[Dict[str, Any]], document: Dict[str, Any]) -> int:
        """
        Store relations in the graph with temporal validity
        
        Args:
            relations: Extracted relations
            document: Source document
            
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        for relation in relations:
            # Create source and target node IDs
            source_id = f"{document['tenant_id']}:{relation['source']['type']}:{relation['source']['text']}"
            target_id = f"{document['tenant_id']}:{relation['target']['type']}:{relation['target']['text']}"
            
            # Create a unique ID for the relation
            relation_id = f"{source_id}:{relation['type']}:{target_id}"
            
            # Set validity window
            t_valid_start = document.get("ts_source") or datetime.now().isoformat()
            
            # Default validity end is far in the future
            t_valid_end = (datetime.now() + timedelta(days=3650)).isoformat()
            
            # Create edge properties
            properties = {
                "type": relation["type"],
                "confidence": relation.get("confidence", 1.0),
                "t_valid_start": t_valid_start,
                "t_valid_end": t_valid_end,
                "tenant_id": document["tenant_id"],
                "provenance": {
                    "document_id": document["id"],
                    "source_tool": document["source_tool"],
                    "ts_extracted": datetime.now().isoformat()
                }
            }
            
            # Check for existing edges
            existing_edges = await self.graph_client.get_edges(
                source_id=source_id,
                target_id=target_id,
                edge_type=relation["type"]
            )
            
            if existing_edges:
                # Handle conflicts
                await self._handle_edge_conflicts(existing_edges, properties, relation_id)
            else:
                # Create new edge
                await self.graph_client.create_edge(
                    edge_id=relation_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=relation["type"],
                    properties=properties
                )
                edges_created += 1
                
        return edges_created
        
    async def _handle_edge_conflicts(self, existing_edges: List[Dict[str, Any]], 
                                   new_properties: Dict[str, Any], 
                                   new_edge_id: str) -> None:
        """
        Handle conflicts between existing edges and a new edge
        
        Args:
            existing_edges: Existing edges
            new_properties: Properties of the new edge
            new_edge_id: ID of the new edge
        """
        # Sort existing edges by validity start time
        sorted_edges = sorted(
            existing_edges,
            key=lambda e: e["properties"]["t_valid_start"]
        )
        
        # Get the new edge's validity window
        new_start = new_properties["t_valid_start"]
        new_end = new_properties["t_valid_end"]
        
        # Check for overlapping edges
        for edge in sorted_edges:
            edge_props = edge["properties"]
            edge_start = edge_props["t_valid_start"]
            edge_end = edge_props["t_valid_end"]
            
            # Check if the new edge overlaps with this edge
            if new_start <= edge_end and new_end >= edge_start:
                # Overlapping validity windows
                
                # Strategy 1: If the new edge has higher confidence, invalidate the old edge
                if new_properties["confidence"] > edge_props.get("confidence", 0):
                    # Update the old edge's validity end to the new edge's start
                    edge_props["t_valid_end"] = new_start
                    await self.graph_client.update_edge(edge["id"], edge_props)
                    
                    # Create the new edge
                    await self.graph_client.create_edge(
                        edge_id=new_edge_id,
                        source_id=edge["source_id"],
                        target_id=edge["target_id"],
                        edge_type=edge["type"],
                        properties=new_properties
                    )
                    
                # Strategy 2: If the new edge has lower confidence, adjust its validity window
                elif new_properties["confidence"] < edge_props.get("confidence", 0):
                    # If the new edge starts before the existing edge
                    if new_start < edge_start:
                        # Truncate the new edge to end before the existing edge starts
                        new_properties["t_valid_end"] = edge_start
                        await self.graph_client.create_edge(
                            edge_id=new_edge_id,
                            source_id=edge["source_id"],
                            target_id=edge["target_id"],
                            edge_type=edge["type"],
                            properties=new_properties
                        )
                    # If the new edge ends after the existing edge
                    elif new_end > edge_end:
                        # Adjust the new edge to start after the existing edge ends
                        new_properties["t_valid_start"] = edge_end
                        await self.graph_client.create_edge(
                            edge_id=f"{new_edge_id}:after",
                            source_id=edge["source_id"],
                            target_id=edge["target_id"],
                            edge_type=edge["type"],
                            properties=new_properties
                        )
                    # Otherwise, the new edge is completely covered by the existing edge
                    # In this case, we don't create the new edge
                    
                # Strategy 3: Equal confidence, use the most recent edge
                else:
                    # Compare extraction timestamps
                    new_ts = new_properties["provenance"]["ts_extracted"]
                    old_ts = edge_props["provenance"].get("ts_extracted")
                    
                    if not old_ts or new_ts > old_ts:
                        # New edge is more recent, replace the old edge
                        edge_props["t_valid_end"] = new_start
                        await self.graph_client.update_edge(edge["id"], edge_props)
                        
                        await self.graph_client.create_edge(
                            edge_id=new_edge_id,
                            source_id=edge["source_id"],
                            target_id=edge["target_id"],
                            edge_type=edge["type"],
                            properties=new_properties
                        )
                        
    async def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents
        
        Args:
            documents: List of normalized documents
            
        Returns:
            List of processing results
        """
        results = []
        
        for document in documents:
            try:
                result = await self.process_document(document)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document {document['id']} in graph sink: {str(e)}")
                # Add error result
                results.append({
                    "document_id": document["id"],
                    "error": str(e)
                })
                
        return results