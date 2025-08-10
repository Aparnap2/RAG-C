"""
MCP (Machine Callable Protocol) implementation for the Universal RAG Stack.
Provides connectivity to MCP servers via JSON-RPC 2.0 over stdio or HTTP+SSE.
"""

from .host import MCPHost
from .transports import StdioTransport, HttpSseTransport, BaseTransport

__all__ = ['MCPHost', 'StdioTransport', 'HttpSseTransport', 'BaseTransport']