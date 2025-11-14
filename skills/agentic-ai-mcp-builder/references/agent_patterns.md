# Agent Design Patterns

Common architectural patterns and best practices for building AI agents.

## Core Agent Patterns

### 1. ReAct Pattern (Reasoning + Acting)

The agent alternates between reasoning about what to do next and taking actions.

```
Thought: What should I do to solve this problem?
Action: Execute a specific function or tool
Observation: Result from the action
... (repeat until complete)
Answer: Final response to user
```

**When to use**: Complex tasks requiring multi-step reasoning and tool use.

**Implementation**:
- Maintain reasoning trace
- Define clear action space
- Parse action/observation loops
- Set max iteration limits

### 2. Chain-of-Thought Pattern

Agent breaks down complex problems into simpler reasoning steps.

**When to use**: Mathematical problems, logical reasoning, multi-step analysis.

**Implementation**:
- Prompt for step-by-step thinking
- Show intermediate steps to user
- Validate each step before proceeding
- Allow backtracking on errors

### 3. Task Decomposition Pattern

Large tasks are broken into smaller sub-tasks that are solved independently.

**When to use**: Complex projects, research tasks, multi-faceted problems.

**Implementation**:
```python
def decompose_task(main_task):
    subtasks = identify_subtasks(main_task)
    results = []
    for subtask in subtasks:
        result = solve_subtask(subtask)
        results.append(result)
    return synthesize_results(results)
```

### 4. Reflection Pattern

Agent evaluates its own outputs and iteratively improves them.

**When to use**: Quality-critical tasks, creative work, code generation.

**Implementation**:
- Generate initial response
- Self-critique the response
- Identify improvements
- Generate improved version
- Repeat until satisfactory

### 5. Multi-Agent Collaboration

Multiple specialized agents work together to solve problems.

**Roles**:
- **Manager Agent**: Coordinates other agents
- **Worker Agents**: Execute specific tasks
- **Critic Agent**: Reviews outputs
- **Synthesizer**: Combines results

**When to use**: Complex domains requiring multiple expertise areas.

### 6. Tool-Using Agent Pattern

Agent has access to external tools/functions and knows when to use them.

**Tool Categories**:
- **Search tools**: Web search, database queries
- **Compute tools**: Calculator, code execution
- **Creative tools**: Image generation, text formatting
- **Communication tools**: Email, messaging

**Implementation**:
```python
tools = {
    "search": search_web,
    "calculate": run_calculation,
    "code_exec": execute_code
}

def use_tool(tool_name, params):
    if tool_name in tools:
        return tools[tool_name](**params)
    return "Tool not found"
```

### 7. Memory-Augmented Agent

Agent maintains and retrieves from a memory store.

**Memory Types**:
- **Short-term**: Current conversation
- **Long-term**: Persistent facts and learnings
- **Episodic**: Past interactions and outcomes
- **Semantic**: General knowledge

**When to use**: Personalization, learning from experience, maintaining context.

### 8. Constitutional AI Pattern

Agent follows explicit rules and principles in decision-making.

**Components**:
- Constitutional principles
- Critique step (check alignment)
- Revision step (fix violations)

**When to use**: Safety-critical applications, regulated domains.

## Agent Communication Patterns

### Request-Response
Simple synchronous communication between user and agent.

```
User → Agent: Request
Agent → User: Response
```

### Publish-Subscribe
Agents subscribe to event streams and react accordingly.

```
Agent A publishes event → Event Bus → Agents B, C, D receive
```

### Message Queue
Asynchronous task processing with queue management.

```
Tasks → Queue → Agent workers process → Results
```

### Blackboard System
Multiple agents contribute to shared knowledge space.

```
Shared Blackboard ↔ Agent 1
                  ↔ Agent 2
                  ↔ Agent 3
```

## Decision-Making Patterns

### Rule-Based Decision
Agent follows if-then rules explicitly.

```python
if condition_a:
    action_1()
elif condition_b:
    action_2()
else:
    default_action()
```

### Probabilistic Decision
Agent weighs options by probability/confidence.

```python
options = [
    {"action": "search", "confidence": 0.8},
    {"action": "calculate", "confidence": 0.6},
    {"action": "ask_clarification", "confidence": 0.3}
]
best_action = max(options, key=lambda x: x["confidence"])
```

### Tree Search Decision
Agent explores decision tree to find optimal path.

```
Root Node
├── Option A
│   ├── Sub-option A1 (score: 0.7)
│   └── Sub-option A2 (score: 0.9) ← Best
└── Option B
    └── Sub-option B1 (score: 0.6)
```

## Error Handling Patterns

### Retry with Exponential Backoff

```python
max_retries = 3
retry_delay = 1

for attempt in range(max_retries):
    try:
        result = perform_action()
        return result
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        time.sleep(retry_delay * (2 ** attempt))
```

### Graceful Degradation

```python
def get_data():
    try:
        return fetch_from_api()
    except APIError:
        return fetch_from_cache()
    except CacheError:
        return get_default_data()
```

### Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failures = 0
        self.threshold = failure_threshold
        self.is_open = False
    
    def call(self, func):
        if self.is_open:
            return "Service unavailable"
        
        try:
            result = func()
            self.failures = 0
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.is_open = True
            raise
```

## Agent State Management

### Stateless Agent
No memory between requests - each request is independent.

**Pros**: Simple, scalable, no state sync issues
**Cons**: No personalization or learning

### Stateful Agent
Maintains conversation history and context.

```python
class StatefulAgent:
    def __init__(self):
        self.conversation_history = []
        self.user_preferences = {}
        self.context = {}
    
    def process(self, user_input):
        # Use history and context
        self.conversation_history.append(user_input)
        response = generate_response(
            user_input, 
            self.conversation_history,
            self.context
        )
        self.conversation_history.append(response)
        return response
```

### Session-Based State
State persists for a session then clears.

```python
class SessionAgent:
    def __init__(self, session_id):
        self.session_id = session_id
        self.state = load_session_state(session_id)
    
    def end_session(self):
        save_session_state(self.session_id, self.state)
        self.state = None
```

## Prompt Engineering Patterns

### Few-Shot Learning
Provide examples in the prompt.

```
Examples:
Input: "What's the weather?"
Output: {action: "check_weather", params: {}}

Input: "Send email to John"
Output: {action: "send_email", params: {recipient: "John"}}

Now process:
Input: "Schedule meeting with Sarah"
```

### Chain Prompting
Break complex tasks into sequential prompts.

```
Prompt 1: "List the key steps to solve this"
Prompt 2: "Execute step 1: {step_1_description}"
Prompt 3: "Execute step 2: {step_2_description}"
...
```

### Structured Output
Request responses in specific formats.

```
Respond in JSON format:
{
  "analysis": "your analysis here",
  "recommendation": "your recommendation",
  "confidence": 0.85
}
```

## Best Practices

### 1. Clear Role Definition
Always define what the agent is and isn't supposed to do.

### 2. Explicit Constraints
List boundaries, limitations, and rules explicitly.

### 3. Observable Behavior
Make agent reasoning visible through logs, traces, or UI.

### 4. Fail-Safe Defaults
Have sensible default behaviors for edge cases.

### 5. Modular Design
Keep agent components loosely coupled and independently testable.

### 6. Versioning
Track agent versions and their behaviors for debugging.

### 7. Testing Strategy
- Unit tests for individual functions
- Integration tests for workflows
- Evaluation sets for quality assurance

### 8. Monitoring
Track key metrics:
- Response time
- Success rate
- Error frequency
- Resource usage
- User satisfaction

## Common Anti-Patterns to Avoid

❌ **God Agent**: One agent trying to do everything
✅ **Solution**: Specialized agents with clear responsibilities

❌ **Brittle Prompts**: Over-specified prompts that break easily
✅ **Solution**: Robust prompts with fallback behaviors

❌ **Ignoring Errors**: Swallowing exceptions silently
✅ **Solution**: Proper error handling and logging

❌ **No Validation**: Trusting all outputs without checking
✅ **Solution**: Validate inputs and outputs

❌ **Infinite Loops**: Agent gets stuck in reasoning loops
✅ **Solution**: Set max iterations and loop detection

❌ **No Human Override**: Agent can't be stopped or corrected
✅ **Solution**: Always allow human intervention

## Agent Evaluation Patterns

### Automated Evaluation

```python
test_cases = [
    {
        "input": "What's 2+2?",
        "expected": "4",
        "criteria": "exact_match"
    },
    {
        "input": "Summarize this article",
        "expected": None,
        "criteria": "length_check",  # Should be < 200 words
    }
]

for test in test_cases:
    result = agent.process(test["input"])
    assert evaluate(result, test["expected"], test["criteria"])
```

### Human-in-the-Loop Evaluation

```python
def human_feedback_loop(agent, task):
    result = agent.process(task)
    feedback = get_human_feedback(result)
    
    if feedback["approved"]:
        return result
    else:
        # Agent learns from feedback
        agent.incorporate_feedback(feedback)
        return agent.process(task)
```

### A/B Testing
Compare two agent versions on same tasks.

```python
def ab_test(agent_a, agent_b, test_set):
    results_a = [agent_a.process(t) for t in test_set]
    results_b = [agent_b.process(t) for t in test_set]
    
    scores_a = [evaluate(r) for r in results_a]
    scores_b = [evaluate(r) for r in results_b]
    
    return {
        "agent_a": mean(scores_a),
        "agent_b": mean(scores_b),
        "winner": "A" if mean(scores_a) > mean(scores_b) else "B"
    }
```

## Scaling Patterns

### Horizontal Scaling
Run multiple agent instances in parallel.

### Caching
Cache common requests and responses.

### Rate Limiting
Prevent overwhelming downstream systems.

### Load Balancing
Distribute requests across agent instances.

```python
class AgentPool:
    def __init__(self, num_agents=5):
        self.agents = [Agent() for _ in range(num_agents)]
        self.current = 0
    
    def process(self, request):
        agent = self.agents[self.current]
        self.current = (self.current + 1) % len(self.agents)
        return agent.process(request)
```
