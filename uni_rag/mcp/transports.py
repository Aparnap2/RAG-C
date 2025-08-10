"""
Transport implementations for MCP connections.
Supports stdio and HTTP+SSE transports with JSON-RPC 2.0 framing.
"""
import json
import asyncio
import logging
import uuid
import subprocess
from typing import Dict, Any, Optional, AsyncGenerator
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseTransport(ABC):
    """Base class for MCP transports"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the transport connection"""
        pass
        
    @abstractmethod
    async def invoke(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a method with parameters"""
        pass
        
    @abstractmethod
    async def subscribe(self, resource: str, params: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to a resource"""
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection"""
        pass


class StdioTransport(BaseTransport):
    """Transport for communicating with MCP servers over stdio"""
    
    def __init__(self, command: str, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.env = env or {}
        self.process = None
        self.request_id = 0
        self.pending_requests = {}
        self._read_task = None
        
    async def initialize(self) -> bool:
        """Start the process and initialize the connection"""
        try:
            # Start the process
            self.process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env
            )
            
            # Start the read loop
            self._read_task = asyncio.create_task(self._read_loop())
            
            # Send initialization message
            response = await self.invoke("mcp.initialize", {
                "version": "1.0",
                "capabilities": ["tools", "resources", "prompts"]
            })
            
            return response.get("status") == "success"
            
        except Exception as e:
            logger.error(f"Failed to initialize stdio transport: {str(e)}")
            return False
            
    async def _read_loop(self):
        """Read and process messages from the process stdout"""
        while self.process and not self.process.stdout.at_eof():
            try:
                # Read a line from stdout
                line = await self.process.stdout.readline()
                if not line:
                    break
                    
                # Parse the JSON-RPC message
                message = json.loads(line.decode('utf-8'))
                
                # Handle the message
                if "id" in message and message["id"] in self.pending_requests:
                    # This is a response to a request
                    request_id = message["id"]
                    future = self.pending_requests.pop(request_id)
                    
                    if "error" in message:
                        future.set_exception(Exception(message["error"].get("message", "Unknown error")))
                    else:
                        future.set_result(message.get("result", {}))
                        
                elif "method" in message:
                    # This is a notification
                    logger.debug(f"Received notification: {message['method']}")
                    # TODO: Handle notifications
                    
            except Exception as e:
                logger.error(f"Error in read loop: {str(e)}")
                
        logger.info("Read loop terminated")
            
    async def invoke(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a method with parameters"""
        if not self.process or self.process.stdin.is_closing():
            raise RuntimeError("Transport not initialized or closed")
            
        # Create a request ID
        self.request_id += 1
        request_id = self.request_id
        
        # Create a future for the response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Create the JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # Send the request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode('utf-8'))
        await self.process.stdin.drain()
        
        # Wait for the response
        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request timed out: {method}")
            
    async def subscribe(self, resource: str, params: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to a resource"""
        # For stdio, we use a special subscription method
        subscription_id = str(uuid.uuid4())
        
        # Start the subscription
        response = await self.invoke("mcp.subscribe", {
            "resource": resource,
            "params": params,
            "subscription_id": subscription_id
        })
        
        if response.get("status") != "success":
            raise RuntimeError(f"Failed to subscribe to {resource}: {response.get('error')}")
            
        # Create a queue for events
        queue = asyncio.Queue()
        
        # Register the subscription
        # TODO: Implement subscription handling
        
        # Yield events from the queue
        try:
            while True:
                event = await queue.get()
                if event is None:  # None is used as a sentinel to end the subscription
                    break
                yield event
        finally:
            # Unsubscribe
            try:
                await self.invoke("mcp.unsubscribe", {
                    "subscription_id": subscription_id
                })
            except Exception as e:
                logger.error(f"Error unsubscribing from {resource}: {str(e)}")
                
    async def close(self) -> None:
        """Close the transport connection"""
        if self.process:
            try:
                # Send a clean shutdown request
                await self.invoke("mcp.shutdown", {})
            except Exception:
                pass
                
            # Cancel the read task
            if self._read_task:
                self._read_task.cancel()
                try:
                    await self._read_task
                except asyncio.CancelledError:
                    pass
                    
            # Terminate the process
            self.process.terminate()
            await self.process.wait()
            self.process = None
            
            # Clear pending requests
            for future in self.pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError("Transport closed"))
            self.pending_requests.clear()


class HttpSseTransport(BaseTransport):
    """Transport for communicating with MCP servers over HTTP+SSE"""
    
    def __init__(self, base_url: str, auth_headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.auth_headers = auth_headers or {}
        self.session = None
        self.request_id = 0
        self.subscriptions = {}
        
    async def initialize(self) -> bool:
        """Initialize the HTTP session"""
        try:
            # Create a session
            self.session = aiohttp.ClientSession(headers=self.auth_headers)
            
            # Send initialization message
            response = await self.invoke("mcp.initialize", {
                "version": "1.0",
                "capabilities": ["tools", "resources", "prompts"]
            })
            
            return response.get("status") == "success"
            
        except Exception as e:
            logger.error(f"Failed to initialize HTTP+SSE transport: {str(e)}")
            if self.session:
                await self.session.close()
                self.session = None
            return False
            
    async def invoke(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a method with parameters"""
        if not self.session:
            raise RuntimeError("Transport not initialized")
            
        # Create a request ID
        self.request_id += 1
        request_id = self.request_id
        
        # Create the JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # Send the request
        async with self.session.post(
            f"{self.base_url}/rpc",
            json=request,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP error: {response.status}")
                
            result = await response.json()
            
            if "error" in result:
                raise RuntimeError(f"RPC error: {result['error'].get('message', 'Unknown error')}")
                
            return result.get("result", {})
            
    async def subscribe(self, resource: str, params: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to a resource using SSE"""
        if not self.session:
            raise RuntimeError("Transport not initialized")
            
        # Create the subscription request
        subscription_id = str(uuid.uuid4())
        request = {
            "resource": resource,
            "params": params,
            "subscription_id": subscription_id
        }
        
        # Start the SSE connection
        response = await self.session.post(
            f"{self.base_url}/subscribe",
            json=request,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            }
        )
        
        if response.status != 200:
            raise RuntimeError(f"HTTP error: {response.status}")
            
        # Process SSE events
        try:
            # Store the last event ID for resuming
            last_event_id = None
            
            # Read the response as a stream
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Parse the SSE event
                if line.startswith("id:"):
                    last_event_id = line[3:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    try:
                        event = json.loads(data)
                        yield event
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in SSE event: {data}")
                elif line.startswith("event:"):
                    # Handle different event types if needed
                    pass
                    
        finally:
            # Unsubscribe
            try:
                await self.invoke("mcp.unsubscribe", {
                    "subscription_id": subscription_id
                })
            except Exception as e:
                logger.error(f"Error unsubscribing from {resource}: {str(e)}")
                
    async def close(self) -> None:
        """Close the transport connection"""
        if self.session:
            await self.session.close()
            self.session = None