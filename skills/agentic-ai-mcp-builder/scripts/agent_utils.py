"""
Agent Utilities - Helper functions for building AI agents

This module provides utility functions to streamline agent development,
including prompt generation, UI creation, and common agent patterns.
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


def create_agent_class(
    agent_name: str,
    system_prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    use_history: bool = True
) -> str:
    """
    Generate boilerplate code for an agent class.
    
    Args:
        agent_name: Name of the agent class
        system_prompt: System prompt for the agent
        tools: List of tool definitions (optional)
        use_history: Whether to maintain conversation history
    
    Returns:
        Python code string for the agent class
    """
    history_code = '''
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.conversation_history = []
        self.tools = TOOLS if TOOLS else []
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
''' if use_history else '''
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.tools = TOOLS if TOOLS else []
'''

    code = f'''"""
{agent_name} - AI Agent Implementation
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json

SYSTEM_PROMPT = """{system_prompt}"""

TOOLS = {json.dumps(tools, indent=4) if tools else None}


class {agent_name}:
    """
    {agent_name} agent for autonomous task execution.
    """
{history_code}
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate agent response.
        
        Args:
            user_input: User's message or query
        
        Returns:
            Dict containing response, reasoning, and metadata
        """
        # Add user message to history
        {"self.add_to_history('user', user_input)" if use_history else "pass"}
        
        # Generate agent response
        response = self._generate_response(user_input)
        
        # Add agent response to history
        {"self.add_to_history('assistant', response['content'])" if use_history else "pass"}
        
        return response
    
    def _generate_response(self, user_input: str) -> Dict[str, Any]:
        """
        Core logic for generating agent responses.
        
        Override this method to implement custom agent behavior.
        """
        # Placeholder - implement actual agent logic here
        return {{
            "content": "Agent response here",
            "reasoning": "Step-by-step reasoning process",
            "confidence": 0.85,
            "actions_taken": [],
            "metadata": {{}}
        }}
    
    def reset(self):
        """Reset agent state."""
        {"self.conversation_history = []" if use_history else "pass"}
'''
    
    return code


def generate_system_prompt(
    role: str,
    capabilities: List[str],
    constraints: List[str],
    output_format: Optional[str] = None,
    examples: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate a comprehensive system prompt for an agent.
    
    Args:
        role: The agent's role/identity
        capabilities: List of things the agent can do
        constraints: List of limitations or rules
        output_format: Desired output format description
        examples: List of example interactions
    
    Returns:
        Formatted system prompt string
    """
    prompt_parts = [f"You are {role}."]
    
    # Add capabilities
    if capabilities:
        prompt_parts.append("\n## Capabilities\n")
        for cap in capabilities:
            prompt_parts.append(f"- {cap}")
    
    # Add constraints
    if constraints:
        prompt_parts.append("\n## Constraints & Guidelines\n")
        for const in constraints:
            prompt_parts.append(f"- {const}")
    
    # Add output format
    if output_format:
        prompt_parts.append(f"\n## Output Format\n{output_format}")
    
    # Add examples
    if examples:
        prompt_parts.append("\n## Examples\n")
        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"\n### Example {i}")
            prompt_parts.append(f"User: {ex.get('input', '')}")
            prompt_parts.append(f"Agent: {ex.get('output', '')}")
    
    return "\n".join(prompt_parts)


def create_agent_workflow(
    steps: List[Dict[str, Any]],
    name: str = "AgentWorkflow"
) -> str:
    """
    Generate code for a multi-step agent workflow.
    
    Args:
        steps: List of workflow steps, each with 'name' and 'description'
        name: Name of the workflow class
    
    Returns:
        Python code for workflow orchestration
    """
    steps_code = "\n        ".join([
        f"self.steps.append({{'name': '{step['name']}', 'description': '{step['description']}', 'status': 'pending'}})"
        for step in steps
    ])
    
    code = f'''class {name}:
    """
    Multi-step agent workflow orchestrator.
    """
    
    def __init__(self):
        self.steps = []
        {steps_code}
        self.current_step = 0
        self.results = {{}}
    
    def execute(self, initial_input: Any) -> Dict[str, Any]:
        """Execute the complete workflow."""
        current_data = initial_input
        
        for i, step in enumerate(self.steps):
            self.current_step = i
            print(f"Executing step {{i+1}}/{{len(self.steps)}}: {{step['name']}}")
            
            step['status'] = 'running'
            
            # Execute step logic here
            result = self._execute_step(step, current_data)
            
            step['status'] = 'complete'
            self.results[step['name']] = result
            current_data = result
        
        return self.results
    
    def _execute_step(self, step: Dict[str, Any], input_data: Any) -> Any:
        """Execute a single workflow step."""
        # Override this method with actual step execution logic
        return input_data
'''
    
    return code


def format_agent_response(
    content: str,
    reasoning: Optional[str] = None,
    confidence: float = 1.0,
    actions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format agent response in a standard structure.
    
    Args:
        content: Main response content
        reasoning: Step-by-step reasoning (optional)
        confidence: Confidence score 0-1
        actions: List of actions taken
        metadata: Additional metadata
    
    Returns:
        Structured response dictionary
    """
    return {
        "content": content,
        "reasoning": reasoning,
        "confidence": confidence,
        "actions_taken": actions or [],
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat()
    }


def validate_agent_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate agent configuration structure.
    
    Args:
        config: Agent configuration dictionary
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_fields = ['name', 'system_prompt', 'capabilities']
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if 'capabilities' in config and not isinstance(config['capabilities'], list):
        errors.append("'capabilities' must be a list")
    
    if 'tools' in config and not isinstance(config['tools'], list):
        errors.append("'tools' must be a list")
    
    return len(errors) == 0, errors


def create_mcp_server_template(
    server_name: str,
    capabilities: List[str],
    resources: List[Dict[str, str]]
) -> str:
    """
    Generate MCP (Model Context Protocol) server template code.
    
    Args:
        server_name: Name of the MCP server
        capabilities: List of server capabilities
        resources: List of resource definitions with 'name' and 'description'
    
    Returns:
        Python code for MCP server
    """
    resources_code = "\n        ".join([
        f"self.resources['{res['name']}'] = {{'description': '{res['description']}', 'handler': None}}"
        for res in resources
    ])
    
    code = f'''"""
{server_name} - MCP Server Implementation
"""

import json
from typing import Dict, Any, List, Optional


class {server_name}:
    """
    Model Context Protocol server for agent integration.
    """
    
    def __init__(self):
        self.name = "{server_name}"
        self.capabilities = {capabilities}
        self.resources = {{}}
        {resources_code}
        self.connections = {{}}
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming MCP requests.
        
        Args:
            request: MCP request with 'method', 'params', and 'id'
        
        Returns:
            MCP response with 'result' or 'error'
        """
        method = request.get('method')
        params = request.get('params', {{}})
        request_id = request.get('id')
        
        try:
            if method == 'initialize':
                result = self._handle_initialize(params)
            elif method == 'get_capabilities':
                result = {{'capabilities': self.capabilities}}
            elif method == 'list_resources':
                result = {{'resources': list(self.resources.keys())}}
            elif method == 'read_resource':
                result = self._handle_read_resource(params)
            else:
                raise ValueError(f"Unknown method: {{method}}")
            
            return {{'id': request_id, 'result': result}}
        
        except Exception as e:
            return {{
                'id': request_id,
                'error': {{'code': -32603, 'message': str(e)}}
            }}
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {{
            'server': self.name,
            'version': '1.0.0',
            'capabilities': self.capabilities
        }}
    
    def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request."""
        resource_name = params.get('name')
        
        if resource_name not in self.resources:
            raise ValueError(f"Unknown resource: {{resource_name}}")
        
        resource = self.resources[resource_name]
        
        # Call resource handler if available
        if resource['handler']:
            content = resource['handler'](params)
        else:
            content = "Resource handler not implemented"
        
        return {{'content': content}}
'''
    
    return code


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Markdown text containing code blocks
    
    Returns:
        List of dicts with 'language' and 'code'
    """
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    return [
        {'language': lang or 'text', 'code': code.strip()}
        for lang, code in matches
    ]


def create_agent_test_suite(agent_name: str, test_cases: List[Dict[str, Any]]) -> str:
    """
    Generate unit test code for an agent.
    
    Args:
        agent_name: Name of the agent class to test
        test_cases: List of test cases with 'name', 'input', and 'expected'
    
    Returns:
        Python unittest code
    """
    test_methods = "\n    ".join([
        f'''def test_{tc['name'].lower().replace(' ', '_')}(self):
        """Test: {tc['name']}"""
        result = self.agent.process("{tc['input']}")
        self.assertIsNotNone(result)
        # Add more specific assertions based on expected output\n'''
        for tc in test_cases
    ])
    
    code = f'''"""
Unit tests for {agent_name}
"""

import unittest
from agent import {agent_name}


class Test{agent_name}(unittest.TestCase):
    """Test suite for {agent_name}."""
    
    def setUp(self):
        """Initialize agent for testing."""
        self.agent = {agent_name}()
    
    {test_methods}


if __name__ == '__main__':
    unittest.main()
'''
    
    return code
