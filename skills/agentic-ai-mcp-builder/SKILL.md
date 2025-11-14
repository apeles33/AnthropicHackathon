---
name: agentic-ai-mcp-builder
description: Builds working AI agents and MCP (Model Context Protocol) integrations across 15 project types including task automation, code review, documentation generation, research assistance, content creation, data analysis, multi-agent systems, and orchestration. Each project produces complete Python agent code with system prompts, example interactions, and interactive demo UIs. Use when building autonomous agents, agent-based systems, MCP servers, or agent orchestration workflows.
---

# Agentic AI & MCP Builder

Build production-ready AI agents and MCP integrations with complete implementations, system prompts, and interactive demos.

## When to Use This Skill

Use this skill when the user wants to:
- Build autonomous AI agents for specific tasks
- Create MCP (Model Context Protocol) integrations
- Develop multi-agent systems or orchestration workflows
- Generate agent system prompts and configurations
- Build agent demo interfaces
- Explore agent design patterns and architectures

## Project Types

This skill supports 15 agent project types:

1. **Task Automation Agent** - Automates repetitive tasks with decision-making
2. **Code Review Agent** - Analyzes code for bugs, style, and best practices
3. **Documentation Generator** - Creates comprehensive technical documentation
4. **Research Assistant** - Gathers, synthesizes, and summarizes information
5. **Content Creator Agent** - Generates articles, posts, and marketing content
6. **Data Analysis Agent** - Analyzes datasets and generates insights
7. **Multi-Agent Debate System** - Multiple agents debate to reach conclusions
8. **Agent Builder (Meta-Agent)** - Builds and configures other agents
9. **File Organizer Agent** - Intelligently organizes files and directories
10. **Meeting Summarizer** - Processes meeting transcripts into summaries
11. **Code Refactoring Agent** - Suggests and implements code improvements
12. **Personal Wiki Agent** - Manages a personal knowledge base
13. **SQL Query Generator** - Creates SQL queries from natural language
14. **API Documentation Agent** - Generates API documentation from code
15. **Workflow Orchestrator** - Coordinates multiple agents and tasks

## Implementation Process

### Step 1: Understand Requirements

Ask the user clarifying questions:
- What specific task should the agent perform?
- What are the expected inputs and outputs?
- Should the agent be interactive or batch-processing?
- Are there specific constraints or requirements?

### Step 2: Load Reference Materials

Before implementing, load relevant reference files:

```python
# Always load agent utilities
view('/mnt/skills/user/agentic-ai-mcp-builder/scripts/agent_utils.py')

# Load patterns based on project type
view('/mnt/skills/user/agentic-ai-mcp-builder/references/agent_patterns.md')

# For MCP integrations
view('/mnt/skills/user/agentic-ai-mcp-builder/references/mcp_patterns.md')

# For project ideas and variations
view('/mnt/skills/user/agentic-ai-mcp-builder/references/project_ideas.md')
```

### Step 3: Build Agent Components

Create the following components for each agent project:

#### 1. Agent Class Implementation
- Main agent class with initialization
- Core decision-making logic
- Tool/function calling capabilities
- State management if needed

#### 2. System Prompt
- Clear role definition
- Capabilities and limitations
- Decision-making guidelines
- Output format specifications
- Example interactions

#### 3. Supporting Functions
- Helper functions for specific tasks
- Data processing utilities
- API interaction code (mock if external APIs needed)
- Error handling

#### 4. Example Interactions
- 3-5 example user inputs
- Expected agent responses
- Edge case handling

#### 5. Demo UI
- Use `assets/agent_template.html` as base
- Interactive interface for testing
- Display agent reasoning/thoughts
- Show conversation history

### Step 4: Generate Complete Project

Create a well-organized project directory:

```
agent_project/
├── agent.py           # Main agent implementation
├── prompts.py         # System prompts and templates
├── utils.py           # Helper functions
├── demo.html          # Interactive demo UI
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── examples.py        # Example usage and tests
```

### Step 5: Create Presentation (Optional)

Generate an HTML presentation showcasing:
- Agent architecture diagram
- Key features and capabilities
- Example interactions with screenshots
- Performance considerations
- Future enhancements

## Agent Design Principles

When building agents, follow these principles:

### 1. Single Responsibility
Each agent should have a clear, focused purpose. Avoid creating "do everything" agents.

### 2. Observable Reasoning
Agents should expose their decision-making process through:
- Thought logs
- Reasoning traces
- Intermediate outputs
- Confidence scores

### 3. Error Handling
Implement robust error handling:
- Graceful degradation
- Clear error messages
- Retry logic where appropriate
- Fallback behaviors

### 4. Testability
Design agents to be easily testable:
- Mock external dependencies
- Provide test fixtures
- Include unit tests
- Document expected behaviors

### 5. Composability
Agents should work well with other agents:
- Clear input/output contracts
- Standard message formats
- Event-driven architecture where applicable

## MCP Integration Guidelines

When building MCP integrations:

1. **Define Clear Protocols** - Specify message formats and exchange patterns
2. **Implement Proper Validation** - Validate all inputs and outputs
3. **Handle Connection Management** - Proper connection lifecycle handling
4. **Support Discovery** - Allow agents to discover each other's capabilities
5. **Enable Monitoring** - Provide observability into agent interactions

## Utility Functions

Use functions from `scripts/agent_utils.py`:

- `create_agent_class()` - Generate agent class boilerplate
- `generate_system_prompt()` - Create system prompts
- `build_demo_ui()` - Generate interactive demo
- `create_agent_workflow()` - Build multi-step workflows
- `format_agent_response()` - Structure agent outputs

## Project Customization

Tailor each project to the user's needs:

1. **Adjust Complexity** - Scale from simple to advanced based on requirements
2. **Add Custom Tools** - Include domain-specific functions
3. **Customize UI** - Modify demo interface for specific use cases
4. **Extend Capabilities** - Add additional features beyond base implementation

## Best Practices

### Code Quality
- Use type hints for clarity
- Write descriptive docstrings
- Follow PEP 8 style guidelines
- Keep functions focused and small

### Documentation
- Explain agent decision-making logic
- Document all configuration options
- Provide clear usage examples
- Include troubleshooting section

### User Experience
- Make demos intuitive and responsive
- Provide helpful error messages
- Show agent progress/status
- Allow easy configuration changes

## Deliverables

Each project should include:

1. ✅ Complete, working Python implementation
2. ✅ Comprehensive system prompts
3. ✅ Interactive HTML demo UI
4. ✅ 3-5 example interactions
5. ✅ README with setup instructions
6. ✅ Requirements.txt with dependencies
7. ✅ Comments explaining key decisions

## Output Format

Present the completed project with:

1. Brief project overview (2-3 sentences)
2. Code files with syntax highlighting
3. Link to demo UI: `computer:///mnt/user-data/outputs/agent_project/demo.html`
4. Quick start instructions
5. Example interaction walkthrough

Keep explanations concise - the code and demo speak for themselves. Focus on enabling the user to immediately run and test their agent.
