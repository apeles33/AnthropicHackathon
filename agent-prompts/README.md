# Agent System Prompts

This directory contains the system prompts for all agents in the meta-builder system.

## Agents

1. **orchestrator.md** - Main coordination agent
2. **builder-agentic.md** - Agentic AI & MCP builder
3. **builder-rag.md** - RAG & ML builder
4. **builder-trading.md** - Trading & Analytics builder
5. **builder-research.md** - Research & Innovation builder

## Usage in Claude Code

These prompts are used in Claude Code's `/agents` tab:

1. Click the **Agents** tab in Claude Code sidebar
2. Click **"+ New Agent"**
3. Enter the agent name (e.g., "Orchestrator" or "builder-agentic")
4. Copy the content from the corresponding `.md` file into the **"System Prompt"** field
5. Configure skills (if applicable):
   - Orchestrator: ALL 4 skills
   - builder-agentic: agentic-ai-mcp-builder skill
   - builder-rag: ai-rag-ml-builder skill
   - builder-trading: stats-trading-analytics-builder skill
   - builder-research: research-paper-builder skill
6. Click **"Create Agent"**

## Agent Interaction Flow

```
User ’ @Orchestrator
  “
  Orchestrator reads all skills
  “
  Orchestrator creates project specification
  “
  Orchestrator spawns @builder-{category}
  “
  Builder reads its domain skill
  “
  Builder generates complete project
  “
  Project saved to output/{category}-{project-name}/
```

## Structure

Each prompt includes:
- **Role definition** - What the agent does
- **Quality standards** - Hackathon-grade output requirements
- **Input specifications** - What the agent expects to receive
- **Output requirements** - What the agent must produce
- **Examples** - Sample interactions
- **Quality checklist** - Self-verification steps

## Testing Agents

After creating all agents:

1. Test Orchestrator:
   ```
   @Orchestrator Generate a test project to verify the system is working
   ```

2. Test individual builders (if needed):
   ```
   @builder-agentic Create a multi-agent debate system
   ```

3. Generate diverse portfolio:
   ```
   @Orchestrator Generate 8 diverse, impressive projects across all 4 categories
   ```

## Tips for Success

- **Be specific**: Give detailed instructions in your requests
- **Use examples**: Reference specific project types from skills
- **Iterate**: If a project isn't perfect, ask the agent to refine it
- **Monitor output**: Check the `output/` directory to see generated projects
- **Review quality**: Open projects in browser/IDE to verify they work
