"""
MCP-driven ingestion with full and incremental sync capabilities.
Implements checkpointing, retries/backoff, and DLQ handling.
"""
import json
import asyncio
import logging
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable
import random

from .mcp.host import MCPHost
from .models import Document

logger = logging.getLogger(__name__)

class QueueClient:
    """
    Abstract queue client interface.
    Implementations could use Kafka, Redis, RabbitMQ, etc.
    """
    async def produce(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        """Produce a message to a topic"""
        raise NotImplementedError("Subclasses must implement produce()")
        
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the queue"""
        raise NotImplementedError("Subclasses must implement health_check()")


class InMemoryQueueClient(QueueClient):
    """
    Simple in-memory queue implementation for testing.
    """
    def __init__(self):
        self.queues = {}
        self.consumers = {}
        
    async def produce(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        """Produce a message to a topic"""
        if topic not in self.queues:
            self.queues[topic] = []
            
        self.queues[topic].append({
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Notify consumers
        if topic in self.consumers:
            for callback in self.consumers[topic]:
                asyncio.create_task(callback(key, value))
                
        return True
        
    async def consume(self, topic: str, callback: Callable[[str, Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a consumer for a topic"""
        if topic not in self.consumers:
            self.consumers[topic] = []
            
        self.consumers[topic].append(callback)
        
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the queue"""
        return {
            "status": "healthy",
            "topics": len(self.queues),
            "consumers": sum(len(consumers) for consumers in self.consumers.values())
        }


class MCPIngestionWorker:
    """
    Worker for ingesting data from MCP tools into the RAG system.
    Supports full and incremental sync with checkpointing.
    """
    def __init__(self, mcp_host: MCPHost, queue_client: QueueClient, config: Dict[str, Any]):
        self.mcp_host = mcp_host
        self.queue_client = queue_client
        self.config = config
        self.checkpoints = config.get("checkpoints", {})
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.retry_backoff = config.get("retry_backoff", 2.0)
        self.retry_jitter = config.get("retry_jitter", 0.1)
        
    async def run_ingestion(self, tool_id: str, tenant_id: str, 
                          params: Optional[Dict[str, Any]] = None, 
                          incremental: bool = False) -> Dict[str, Any]:
        """
        Run ingestion using an MCP tool
        
        Args:
            tool_id: MCP tool identifier
            tenant_id: Tenant identifier
            params: Optional parameters for the tool
            incremental: Whether to use incremental sync with cursor
            
        Returns:
            Result with items_processed and cursor
        """
        # Set up parameters
        params = params or {}
        if incremental and tool_id in self.checkpoints:
            params["cursor"] = self.checkpoints.get(tool_id, {}).get("cursor")
            
        # Initialize retry counter
        retry_count = 0
        last_error = None
        
        # Retry loop
        while retry_count <= self.max_retries:
            try:
                # Invoke MCP tool
                response = await self.mcp_host.invoke_tool(
                    tool_id=tool_id,
                    params=params,
                    tenant_id=tenant_id
                )
                
                # Process items
                items_processed = 0
                for item in response.get("items", []):
                    # Normalize item
                    document = self._normalize_to_document(item, tool_id, tenant_id)
                    
                    # Compute checksum
                    checksum = self._compute_checksum(document)
                    document["checksum"] = checksum
                    
                    # Produce to queue with idempotency key
                    await self.queue_client.produce(
                        topic="ingestion",
                        key=f"{tenant_id}:{document['source_id']}",
                        value=document
                    )
                    items_processed += 1
                    
                # Store checkpoint if provided
                if "cursor" in response:
                    if tool_id not in self.checkpoints:
                        self.checkpoints[tool_id] = {}
                    self.checkpoints[tool_id]["cursor"] = response["cursor"]
                    self.checkpoints[tool_id]["last_sync"] = datetime.now().isoformat()
                    await self._save_checkpoints()
                    
                return {
                    "items_processed": items_processed,
                    "cursor": response.get("cursor")
                }
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    # Calculate backoff with jitter
                    delay = self.retry_delay * (self.retry_backoff ** (retry_count - 1))
                    jitter = random.uniform(-self.retry_jitter, self.retry_jitter) * delay
                    delay = max(0, delay + jitter)
                    
                    logger.warning(f"Ingestion failed for {tool_id}, retry {retry_count}/{self.max_retries} in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded, send to DLQ
                    logger.error(f"Ingestion failed for {tool_id} after {self.max_retries} retries: {str(e)}")
                    await self._send_to_dlq(tool_id, tenant_id, params, str(e))
                    
        # If we get here, all retries failed
        raise last_error or RuntimeError(f"Ingestion failed for {tool_id}")
            
    async def _send_to_dlq(self, tool_id: str, tenant_id: str, params: Dict[str, Any], error: str) -> None:
        """Send a failed ingestion to the dead letter queue"""
        await self.queue_client.produce(
            topic="ingestion_dlq",
            key=f"{tenant_id}:{tool_id}",
            value={
                "tool_id": tool_id,
                "tenant_id": tenant_id,
                "params": params,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "retry_count": self.max_retries
            }
        )
        
    async def _save_checkpoints(self) -> None:
        """Save checkpoints to persistent storage"""
        # In a real implementation, this would save to a database or file
        # For now, just log the checkpoints
        logger.info(f"Saving checkpoints: {json.dumps(self.checkpoints)}")
        
        # Update the config
        self.config["checkpoints"] = self.checkpoints
        
    def _normalize_to_document(self, item: Dict[str, Any], tool_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Normalize an item from an MCP tool into a canonical document format
        
        Args:
            item: Raw item from MCP tool
            tool_id: MCP tool identifier
            tenant_id: Tenant identifier
            
        Returns:
            Normalized document
        """
        # Extract source ID
        source_id = item.get("id") or item.get("source_id")
        if not source_id:
            # Generate a deterministic ID if none is provided
            source_id = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
            
        # Extract content
        content = item.get("content") or item.get("text") or ""
        
        # Extract metadata
        metadata = item.get("metadata") or {}
        
        # Extract ACLs
        acl = item.get("acl") or []
        
        # Extract timestamps
        ts_source = item.get("timestamp") or item.get("created_at") or datetime.now().isoformat()
        ts_ingested = datetime.now().isoformat()
        
        # Create the document
        document = {
            "id": f"{tenant_id}:{tool_id}:{source_id}",
            "tenant_id": tenant_id,
            "source_tool": tool_id,
            "source_id": source_id,
            "content": content,
            "metadata": metadata,
            "acl": acl,
            "ts_source": ts_source,
            "ts_ingested": ts_ingested,
            "schema_version": "1.0"
        }
        
        return document
        
    def _compute_checksum(self, document: Dict[str, Any]) -> str:
        """
        Compute a checksum for a document for idempotency
        
        Args:
            document: Document to checksum
            
        Returns:
            Checksum string
        """
        # Create a copy with only the fields we want to include in the checksum
        checksum_doc = {
            "source_id": document["source_id"],
            "content": document["content"],
            "metadata": document["metadata"],
            "ts_source": document["ts_source"]
        }
        
        # Compute MD5 hash
        return hashlib.md5(json.dumps(checksum_doc, sort_keys=True).encode()).hexdigest()
        
    async def process_event(self, tool_id: str, data: Dict[str, Any], tenant_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an event from the API
        
        Args:
            tool_id: MCP tool identifier
            data: Event data
            tenant_id: Tenant identifier
            user_id: Optional user identifier
            
        Returns:
            Processing result
        """
        # Normalize the event
        document = self._normalize_to_document(data, tool_id, tenant_id)
        
        # Compute checksum
        checksum = self._compute_checksum(document)
        document["checksum"] = checksum
        
        # Produce to queue with idempotency key
        await self.queue_client.produce(
            topic="ingestion",
            key=f"{tenant_id}:{document['source_id']}",
            value=document
        )
        
        return {
            "items_processed": 1,
            "document_id": document["id"]
        }
        
    async def start_streaming_ingestion(self, resource_id: str, tenant_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Start streaming ingestion from an MCP resource
        
        Args:
            resource_id: MCP resource identifier
            tenant_id: Tenant identifier
            params: Optional parameters for the resource
        """
        params = params or {}
        
        # Get the last event ID if available
        if resource_id in self.checkpoints:
            params["last_event_id"] = self.checkpoints.get(resource_id, {}).get("last_event_id")
            
        try:
            # Subscribe to the resource
            async for event in self.mcp_host.subscribe_resource(
                resource_id=resource_id,
                params=params,
                tenant_id=tenant_id
            ):
                try:
                    # Process the event
                    document = self._normalize_to_document(event["data"], resource_id, tenant_id)
                    
                    # Compute checksum
                    checksum = self._compute_checksum(document)
                    document["checksum"] = checksum
                    
                    # Produce to queue with idempotency key
                    await self.queue_client.produce(
                        topic="ingestion",
                        key=f"{tenant_id}:{document['source_id']}",
                        value=document
                    )
                    
                    # Store the event ID
                    if "id" in event:
                        if resource_id not in self.checkpoints:
                            self.checkpoints[resource_id] = {}
                        self.checkpoints[resource_id]["last_event_id"] = event["id"]
                        self.checkpoints[resource_id]["last_event"] = datetime.now().isoformat()
                        
                        # Periodically save checkpoints
                        if random.random() < 0.1:  # Save roughly every 10 events
                            await self._save_checkpoints()
                            
                except Exception as e:
                    logger.error(f"Error processing streaming event from {resource_id}: {str(e)}")
                    # Send to DLQ
                    await self.queue_client.produce(
                        topic="ingestion_dlq",
                        key=f"{tenant_id}:{resource_id}",
                        value={
                            "resource_id": resource_id,
                            "tenant_id": tenant_id,
                            "event": event,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Streaming ingestion failed for {resource_id}: {str(e)}")
            # Save checkpoint before exiting
            await self._save_checkpoints()