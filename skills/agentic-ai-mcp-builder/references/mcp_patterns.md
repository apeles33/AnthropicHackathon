# MCP (Model Context Protocol) Integration Patterns

Comprehensive guide to building Model Context Protocol integrations for agent systems.

## What is MCP?

Model Context Protocol (MCP) is a standardized protocol for AI agents to communicate with:
- External data sources
- Tools and services  
- Other agents
- Computing resources

MCP provides a unified interface for context sharing and tool invocation across AI systems.

## Core MCP Concepts

### Resources
Data sources that agents can read from:
- Files and documents
- Database tables
- API endpoints
- Real-time data streams

### Tools
Functions agents can invoke:
- Calculations
- Data transformations
- External service calls
- System operations

### Prompts
Pre-defined prompt templates with parameters:
- Task-specific instructions
- Domain expertise
- Workflow templates

### Sampling
Request completions from other models or agents.

## MCP Server Patterns

### 1. Basic MCP Server

```python
class BasicMCPServer:
    """
    Minimal MCP server implementation.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.resources = {}
        self.tools = {}
        self.prompts = {}
    
    def register_resource(self, name: str, handler: Callable):
        """Register a resource with its handler function."""
        self.resources[name] = handler
    
    def register_tool(self, name: str, handler: Callable, schema: Dict):
        """Register a tool with schema and handler."""
        self.tools[name] = {
            "handler": handler,
            "schema": schema
        }
    
    def handle_request(self, method: str, params: Dict) -> Dict:
        """Handle incoming MCP requests."""
        if method == "list_resources":
            return {"resources": list(self.resources.keys())}
        
        elif method == "read_resource":
            resource_name = params["name"]
            handler = self.resources.get(resource_name)
            if handler:
                return {"content": handler(params)}
            return {"error": "Resource not found"}
        
        elif method == "list_tools":
            return {"tools": [
                {"name": name, "schema": tool["schema"]}
                for name, tool in self.tools.items()
            ]}
        
        elif method == "call_tool":
            tool_name = params["name"]
            tool = self.tools.get(tool_name)
            if tool:
                result = tool["handler"](params["arguments"])
                return {"result": result}
            return {"error": "Tool not found"}
        
        return {"error": "Unknown method"}
```

### 2. File System MCP Server

```python
import os
import json
from pathlib import Path

class FileSystemMCPServer:
    """
    MCP server for file system access.
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up resource and tool handlers."""
        self.resources = {
            "file": self.read_file,
            "directory": self.list_directory
        }
        
        self.tools = {
            "search_files": {
                "schema": {
                    "name": "search_files",
                    "description": "Search for files matching pattern",
                    "parameters": {
                        "pattern": {"type": "string", "required": True},
                        "recursive": {"type": "boolean", "default": True}
                    }
                },
                "handler": self.search_files
            },
            "create_file": {
                "schema": {
                    "name": "create_file",
                    "description": "Create a new file",
                    "parameters": {
                        "path": {"type": "string", "required": True},
                        "content": {"type": "string", "required": True}
                    }
                },
                "handler": self.create_file
            }
        }
    
    def read_file(self, params: Dict) -> str:
        """Read file contents."""
        file_path = self.base_path / params["path"]
        if not file_path.is_file():
            raise ValueError(f"File not found: {params['path']}")
        return file_path.read_text()
    
    def list_directory(self, params: Dict) -> List[str]:
        """List directory contents."""
        dir_path = self.base_path / params.get("path", "")
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {params.get('path')}")
        return [str(p.relative_to(self.base_path)) for p in dir_path.iterdir()]
    
    def search_files(self, params: Dict) -> List[str]:
        """Search for files matching pattern."""
        pattern = params["pattern"]
        recursive = params.get("recursive", True)
        
        if recursive:
            matches = self.base_path.rglob(pattern)
        else:
            matches = self.base_path.glob(pattern)
        
        return [str(p.relative_to(self.base_path)) for p in matches]
    
    def create_file(self, params: Dict) -> Dict[str, str]:
        """Create a new file."""
        file_path = self.base_path / params["path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(params["content"])
        return {"status": "created", "path": str(file_path)}
```

### 3. Database MCP Server

```python
class DatabaseMCPServer:
    """
    MCP server for database access.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.schema = self.load_schema()
    
    def load_schema(self) -> Dict:
        """Load database schema information."""
        # Mock implementation
        return {
            "tables": {
                "users": {
                    "columns": ["id", "name", "email", "created_at"],
                    "primary_key": "id"
                },
                "orders": {
                    "columns": ["id", "user_id", "amount", "status"],
                    "primary_key": "id"
                }
            }
        }
    
    def handle_request(self, method: str, params: Dict) -> Dict:
        """Handle MCP requests."""
        if method == "list_resources":
            return {
                "resources": [
                    {"name": "schema", "type": "database_schema"},
                    {"name": "tables", "type": "table_list"}
                ]
            }
        
        elif method == "read_resource":
            if params["name"] == "schema":
                return {"content": json.dumps(self.schema, indent=2)}
            elif params["name"] == "tables":
                return {"content": list(self.schema["tables"].keys())}
        
        elif method == "call_tool":
            tool = params["name"]
            args = params["arguments"]
            
            if tool == "execute_query":
                return {"result": self.execute_query(args["sql"])}
            elif tool == "get_table_info":
                return {"result": self.schema["tables"].get(args["table"])}
        
        return {"error": "Unknown method"}
    
    def execute_query(self, sql: str) -> List[Dict]:
        """Execute SQL query."""
        # Mock implementation
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ]
```

### 4. API MCP Server

```python
import requests
from typing import Dict, Any, List

class APIMCPServer:
    """
    MCP server for external API access.
    """
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.endpoints = self.discover_endpoints()
    
    def discover_endpoints(self) -> Dict[str, Dict]:
        """Discover available API endpoints."""
        # Mock implementation
        return {
            "get_user": {
                "method": "GET",
                "path": "/users/{id}",
                "params": {"id": "string"}
            },
            "create_user": {
                "method": "POST",
                "path": "/users",
                "body": {"name": "string", "email": "string"}
            },
            "list_users": {
                "method": "GET",
                "path": "/users",
                "params": {"limit": "number", "offset": "number"}
            }
        }
    
    def handle_request(self, method: str, params: Dict) -> Dict:
        """Handle MCP requests."""
        if method == "list_tools":
            return {
                "tools": [
                    {
                        "name": endpoint_name,
                        "description": f"API endpoint: {config['method']} {config['path']}",
                        "schema": config
                    }
                    for endpoint_name, config in self.endpoints.items()
                ]
            }
        
        elif method == "call_tool":
            endpoint_name = params["name"]
            args = params["arguments"]
            return {"result": self.call_endpoint(endpoint_name, args)}
        
        return {"error": "Unknown method"}
    
    def call_endpoint(self, endpoint_name: str, args: Dict) -> Any:
        """Call an API endpoint."""
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        method = endpoint["method"]
        path = endpoint["path"].format(**args)
        url = f"{self.base_url}{path}"
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if method == "GET":
            response = requests.get(url, params=args.get("params", {}), headers=headers)
        elif method == "POST":
            response = requests.post(url, json=args.get("body", {}), headers=headers)
        elif method == "PUT":
            response = requests.put(url, json=args.get("body", {}), headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        return response.json()
```

## MCP Client Patterns

### 1. Basic MCP Client

```python
class MCPClient:
    """
    Client for connecting to MCP servers.
    """
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.capabilities = None
    
    def connect(self):
        """Initialize connection to MCP server."""
        response = self.send_request("initialize", {})
        self.capabilities = response.get("capabilities", [])
        return self.capabilities
    
    def send_request(self, method: str, params: Dict) -> Dict:
        """Send a request to the MCP server."""
        # Mock implementation - replace with actual network call
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.generate_request_id()
        }
        
        # Simulate response
        return {"result": {}, "id": request["id"]}
    
    def list_resources(self) -> List[str]:
        """List available resources."""
        response = self.send_request("list_resources", {})
        return response["result"]["resources"]
    
    def read_resource(self, name: str, params: Dict = None) -> str:
        """Read a resource."""
        params = params or {}
        params["name"] = name
        response = self.send_request("read_resource", params)
        return response["result"]["content"]
    
    def list_tools(self) -> List[Dict]:
        """List available tools."""
        response = self.send_request("list_tools", {})
        return response["result"]["tools"]
    
    def call_tool(self, name: str, arguments: Dict) -> Any:
        """Call a tool."""
        response = self.send_request("call_tool", {
            "name": name,
            "arguments": arguments
        })
        return response["result"]["result"]
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())
```

### 2. Agent with MCP Integration

```python
class MCPAgent:
    """
    AI agent that uses MCP to access external resources and tools.
    """
    
    def __init__(self, mcp_servers: List[MCPClient]):
        self.mcp_servers = mcp_servers
        self.available_tools = {}
        self.available_resources = {}
        self.discover_capabilities()
    
    def discover_capabilities(self):
        """Discover capabilities from all MCP servers."""
        for server in self.mcp_servers:
            try:
                server.connect()
                
                # Discover tools
                tools = server.list_tools()
                for tool in tools:
                    self.available_tools[tool["name"]] = {
                        "server": server,
                        "schema": tool["schema"]
                    }
                
                # Discover resources
                resources = server.list_resources()
                for resource in resources:
                    self.available_resources[resource] = server
            
            except Exception as e:
                print(f"Error connecting to server: {e}")
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """Process user input using available MCP capabilities."""
        # Determine what tools/resources are needed
        required_tools = self.identify_required_tools(user_input)
        required_resources = self.identify_required_resources(user_input)
        
        # Gather information from resources
        context = {}
        for resource_name in required_resources:
            server = self.available_resources.get(resource_name)
            if server:
                context[resource_name] = server.read_resource(resource_name)
        
        # Execute required tools
        tool_results = {}
        for tool_name, args in required_tools.items():
            tool_info = self.available_tools.get(tool_name)
            if tool_info:
                result = tool_info["server"].call_tool(tool_name, args)
                tool_results[tool_name] = result
        
        # Generate response using context and tool results
        response = self.generate_response(user_input, context, tool_results)
        
        return {
            "content": response,
            "context_used": list(context.keys()),
            "tools_used": list(tool_results.keys())
        }
    
    def identify_required_tools(self, user_input: str) -> Dict[str, Dict]:
        """Identify which tools are needed for this input."""
        # Simplified - in practice would use LLM reasoning
        return {}
    
    def identify_required_resources(self, user_input: str) -> List[str]:
        """Identify which resources are needed."""
        # Simplified - in practice would use LLM reasoning
        return []
    
    def generate_response(self, user_input: str, context: Dict, tool_results: Dict) -> str:
        """Generate final response."""
        return "Agent response incorporating MCP data"
```

## MCP Message Format

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "call_tool",
  "params": {
    "name": "search_files",
    "arguments": {
      "pattern": "*.py",
      "recursive": true
    }
  },
  "id": "req-123"
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "result": {
    "files": ["agent.py", "utils.py", "main.py"]
  },
  "id": "req-123"
}
```

### Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid parameters",
    "data": {
      "param": "pattern",
      "reason": "Required parameter missing"
    }
  },
  "id": "req-123"
}
```

## MCP Integration Patterns

### 1. Resource Caching

```python
class CachingMCPClient(MCPClient):
    """MCP client with caching support."""
    
    def __init__(self, server_url: str, cache_ttl: int = 300):
        super().__init__(server_url)
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def read_resource(self, name: str, params: Dict = None) -> str:
        """Read resource with caching."""
        cache_key = f"{name}:{json.dumps(params)}"
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                return entry["content"]
        
        content = super().read_resource(name, params)
        
        self.cache[cache_key] = {
            "content": content,
            "timestamp": time.time()
        }
        
        return content
```

### 2. Connection Pooling

```python
class MCPConnectionPool:
    """Pool of MCP client connections."""
    
    def __init__(self, server_url: str, pool_size: int = 5):
        self.server_url = server_url
        self.pool = [MCPClient(server_url) for _ in range(pool_size)]
        self.available = self.pool.copy()
        self.lock = threading.Lock()
    
    def acquire(self) -> MCPClient:
        """Get a client from the pool."""
        with self.lock:
            if self.available:
                return self.available.pop()
            # Create new client if pool empty
            return MCPClient(self.server_url)
    
    def release(self, client: MCPClient):
        """Return client to pool."""
        with self.lock:
            if len(self.available) < len(self.pool):
                self.available.append(client)
```

### 3. Async MCP Client

```python
import asyncio
import aiohttp

class AsyncMCPClient:
    """Asynchronous MCP client."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
    
    async def connect(self):
        """Initialize async connection."""
        self.session = aiohttp.ClientSession()
        await self.send_request("initialize", {})
    
    async def send_request(self, method: str, params: Dict) -> Dict:
        """Send async request."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(uuid.uuid4())
        }
        
        async with self.session.post(self.server_url, json=request) as response:
            return await response.json()
    
    async def call_tool(self, name: str, arguments: Dict) -> Any:
        """Call tool asynchronously."""
        response = await self.send_request("call_tool", {
            "name": name,
            "arguments": arguments
        })
        return response["result"]["result"]
    
    async def close(self):
        """Close connection."""
        if self.session:
            await self.session.close()
```

## Best Practices

### 1. Resource Discovery
Allow agents to discover available resources dynamically.

### 2. Schema Validation
Validate all requests and responses against schemas.

### 3. Error Handling
Provide clear error messages with actionable information.

### 4. Versioning
Support multiple protocol versions for backward compatibility.

### 5. Security
- Authenticate all requests
- Validate permissions
- Sanitize inputs
- Rate limiting

### 6. Monitoring
Track:
- Request/response times
- Error rates
- Resource usage
- Tool invocation patterns

### 7. Documentation
Document all:
- Available resources
- Tool schemas
- Example usage
- Error codes

## Common Use Cases

### Multi-Agent Collaboration

```python
# Agent 1: Research agent with web search
research_server = APIMCPServer("https://search-api.example.com")

# Agent 2: Data analysis agent with database
data_server = DatabaseMCPServer("postgresql://...")

# Agent 3: Orchestrator that coordinates both
orchestrator = MCPAgent([
    MCPClient.from_server(research_server),
    MCPClient.from_server(data_server)
])

result = orchestrator.process("Find latest trends and analyze our customer data")
```

### RAG (Retrieval-Augmented Generation)

```python
# MCP server provides document retrieval
doc_server = FileSystemMCPServer("/docs")

# Agent uses MCP to retrieve relevant docs
agent = MCPAgent([MCPClient.from_server(doc_server)])
response = agent.process("Explain our refund policy")
# Agent retrieves relevant policy documents via MCP
```

### Tool Composition

```python
# Chain multiple tools together
client = MCPClient("https://mcp-server.example.com")

# Search files
files = client.call_tool("search_files", {"pattern": "*.log"})

# Read each file
for file in files:
    content = client.read_resource("file", {"path": file})
    # Process content
```
