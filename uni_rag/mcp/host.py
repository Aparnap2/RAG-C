"""
MCP Host implementation for connecting to MCP servers via JSON-RPC 2.0.
Supports both stdio and HTTP+SSE transports.
"""
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from .transports import StdioTransport, HttpSseTransport, BaseTransport

logger = logging.getLogger(__name__)

class MCPHost:
    """
    Host for connecting to MCP servers and discovering/invoking tools, resources, and prompts.
    """
    def __init__(self, config: Dict[str, Any]):
        self.servers: Dict[str, BaseTransport] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.config = config
        self.audit_logger = self._setup_audit_logger()
        
    def _setup_audit_logger(self):
        """Set up a dedicated logger for audit records"""
        audit_logger = logging.getLogger("mcp.audit")
        # Configure audit logger based on config
        return audit_logger
        
    async def connect_server(self, server_id: str, transport_type: str, connection_params: Dict[str, Any]) -> bool:
        """
        Connect to an MCP server via stdio or HTTP+SSE
        
        Args:
            server_id: Unique identifier for the server
            transport_type: "stdio" or "http+sse"
            connection_params: Parameters for the transport
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if transport_type == "stdio":
                self.servers[server_id] = StdioTransport(**connection_params)
            elif transport_type == "http+sse":
                self.servers[server_id] = HttpSseTransport(**connection_params)
            else:
                raise ValueError(f"Unsupported transport type: {transport_type}")
                
            # Initialize the connection
            await self.servers[server_id].initialize()
            
            # Discover tools, resources, and prompts
            await self.discover_capabilities(server_id)
            
            logger.info(f"Connected to MCP server {server_id} via {transport_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_id}: {str(e)}")
            return False
        
    async def discover_capabilities(self, server_id: str) -> Tuple[int, int, int]:
        """
        Discover tools, resources, and prompts from an MCP server
        
        Args:
            server_id: Server identifier
            
        Returns:
            Tuple of (tool_count, resource_count, prompt_count)
        """
        server = self.servers[server_id]
        tool_count, resource_count, prompt_count = 0, 0, 0
        
        # List tools
        try:
            tools_response = await server.invoke("mcp.list_tools", {})
            for tool in tools_response.get("tools", []):
                self.tools[f"{server_id}.{tool['name']}"] = {
                    "server_id": server_id,
                    "schema": tool.get("schema", {}),
                    "description": tool.get("description", ""),
                    "permissions": tool.get("permissions", [])
                }
                tool_count += 1
        except Exception as e:
            logger.error(f"Failed to discover tools from server {server_id}: {str(e)}")
            
        # List resources
        try:
            resources_response = await server.invoke("mcp.list_resources", {})
            for resource in resources_response.get("resources", []):
                self.resources[f"{server_id}.{resource['name']}"] = {
                    "server_id": server_id,
                    "schema": resource.get("schema", {}),
                    "description": resource.get("description", "")
                }
                resource_count += 1
        except Exception as e:
            logger.error(f"Failed to discover resources from server {server_id}: {str(e)}")
            
        # List prompts
        try:
            prompts_response = await server.invoke("mcp.list_prompts", {})
            for prompt in prompts_response.get("prompts", []):
                self.prompts[f"{server_id}.{prompt['name']}"] = {
                    "server_id": server_id,
                    "template": prompt.get("template", ""),
                    "description": prompt.get("description", "")
                }
                prompt_count += 1
        except Exception as e:
            logger.error(f"Failed to discover prompts from server {server_id}: {str(e)}")
            
        logger.info(f"Discovered {tool_count} tools, {resource_count} resources, and {prompt_count} prompts from server {server_id}")
        return (tool_count, resource_count, prompt_count)
            
    def _validate_params(self, params: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate parameters against a JSON schema
        
        Args:
            params: Parameters to validate
            schema: JSON schema
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Implement JSON schema validation
        # For now, just return True
        return True
        
    def _check_permissions(self, tool_id: str, tenant_id: str, user_id: Optional[str] = None) -> bool:
        """
        Check if the tenant/user has permission to use the tool
        
        Args:
            tool_id: Tool identifier
            tenant_id: Tenant identifier
            user_id: Optional user identifier
            
        Returns:
            True if permitted, raises ValueError otherwise
        """
        # Check tenant allow-list
        tenant_tools = self.config.get("tenants", {}).get(tenant_id, {}).get("allowed_tools", [])
        if not tenant_tools or tool_id not in tenant_tools:
            raise ValueError(f"Tool {tool_id} not allowed for tenant {tenant_id}")
            
        # Check user permissions if provided
        if user_id:
            user_tools = self.config.get("tenants", {}).get(tenant_id, {}).get("users", {}).get(user_id, {}).get("allowed_tools", [])
            if user_tools and tool_id not in user_tools:
                raise ValueError(f"Tool {tool_id} not allowed for user {user_id}")
                
        return True
        
    def _log_invocation(self, tool_id: str, params: Dict[str, Any], tenant_id: Optional[str], user_id: Optional[str]) -> str:
        """
        Log tool invocation for audit
        
        Args:
            tool_id: Tool identifier
            params: Parameters
            tenant_id: Optional tenant identifier
            user_id: Optional user identifier
            
        Returns:
            Invocation ID
        """
        invocation_id = str(uuid.uuid4())
        
        # Log to audit logger
        self.audit_logger.info(
            "tool_invocation",
            extra={
                "invocation_id": invocation_id,
                "tool_id": tool_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "params": json.dumps(params)
            }
        )
        
        return invocation_id
            
    async def invoke_tool(self, tool_id: str, params: Dict[str, Any], 
                         tenant_id: Optional[str] = None, 
                         user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke an MCP tool with parameters
        
        Args:
            tool_id: Tool identifier (server_id.tool_name)
            params: Parameters for the tool
            tenant_id: Optional tenant identifier for permission checks
            user_id: Optional user identifier for permission checks
            
        Returns:
            Tool response
        """
        tool = self.tools.get(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")
            
        # Validate params against schema
        self._validate_params(params, tool["schema"])
        
        # Check permissions
        if tenant_id:
            self._check_permissions(tool_id, tenant_id, user_id)
            
        # Invoke tool
        server = self.servers[tool["server_id"]]
        tool_name = tool_id.split(".", 1)[1]
        
        # Log invocation for audit
        invocation_id = self._log_invocation(tool_id, params, tenant_id, user_id)
        
        try:
            result = await server.invoke(tool_name, params)
            
            # Log success
            self.audit_logger.info(
                "tool_success",
                extra={
                    "invocation_id": invocation_id,
                    "result_size": len(json.dumps(result))
                }
            )
            
            return result
            
        except Exception as e:
            # Log error
            self.audit_logger.error(
                "tool_error",
                extra={
                    "invocation_id": invocation_id,
                    "error": str(e)
                }
            )
            raise
            
    async def subscribe_resource(self, resource_id: str, params: Dict[str, Any],
                               tenant_id: Optional[str] = None,
                               user_id: Optional[str] = None):
        """
        Subscribe to an MCP resource
        
        Args:
            resource_id: Resource identifier (server_id.resource_name)
            params: Parameters for the subscription
            tenant_id: Optional tenant identifier for permission checks
            user_id: Optional user identifier for permission checks
            
        Returns:
            Async generator yielding events
        """
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource not found: {resource_id}")
            
        # Validate params against schema
        self._validate_params(params, resource["schema"])
        
        # Check permissions
        if tenant_id:
            self._check_permissions(resource_id, tenant_id, user_id)
            
        # Subscribe to resource
        server = self.servers[resource["server_id"]]
        resource_name = resource_id.split(".", 1)[1]
        
        # Log subscription for audit
        invocation_id = self._log_invocation(resource_id, params, tenant_id, user_id)
        
        try:
            async for event in server.subscribe(resource_name, params):
                yield event
                
        except Exception as e:
            # Log error
            self.audit_logger.error(
                "resource_error",
                extra={
                    "invocation_id": invocation_id,
                    "error": str(e)
                }
            )
            raise
            
    async def get_prompt(self, prompt_id: str, params: Dict[str, Any]) -> str:
        """
        Get a prompt template and fill it with parameters
        
        Args:
            prompt_id: Prompt identifier (server_id.prompt_name)
            params: Parameters to fill in the prompt
            
        Returns:
            Filled prompt template
        """
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_id}")
            
        # Get the template
        template = prompt["template"]
        
        # Fill in parameters
        for key, value in params.items():
            template = template.replace(f"{{{key}}}", str(value))
            
        return template
        
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of all connected MCP servers
        
        Returns:
            Health status for each server
        """
        status = {}
        
        for server_id, server in self.servers.items():
            try:
                # Try to ping the server
                await server.invoke("mcp.ping", {})
                status[server_id] = "healthy"
            except Exception as e:
                status[server_id] = f"unhealthy: {str(e)}"
                
        return {
            "status": "healthy" if all(s == "healthy" for s in status.values()) else "unhealthy",
            "servers": status
        }