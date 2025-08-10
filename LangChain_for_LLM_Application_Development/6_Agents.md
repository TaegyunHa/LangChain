# Agents

## Key LangChain Objects

- `langchain.agents.load_tools`
- `langchain.agents.initialize_agent`
- `langchain.agents.AgentType`
- `langchain.agents.agent_toolkits.create_python_agent`
- `langchain.tools.python.tool.PythonREPLTool`
- `langchain.agents.tool`

## Overview

Agents are systems that use LLMs to determine which actions to take and in what order. Unlike chains with predefined sequences, agents dynamically decide their next steps based on user input and intermediate results.

**Agent Components**:
1. **LLM**: The language model that powers decision-making
2. **Tools**: Functions the agent can call to perform specific tasks
3. **Agent Type**: The reasoning strategy (e.g., ReAct, Zero-shot)
4. **Memory**: Optional storage for conversation history

## Built-in LangChain Tools

### Loading Pre-built Tools

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Load built-in tools
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

# Create agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)
```

**Common Built-in Tools**:
- **llm-math**: Performs mathematical calculations
- **wikipedia**: Searches Wikipedia for information
- **python_repl**: Executes Python code
- **requests**: Makes HTTP requests
- **search**: Web search capabilities

### Mathematical Tool Example

```python
agent("What is 25% of 300?")
```

**Agent Reasoning Process**:
1. Analyzes the question to identify it requires math
2. Selects the llm-math tool
3. Performs the calculation: 300 * 0.25 = 75
4. Returns the result with explanation

### Wikipedia Tool Example

```python
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"

result = agent(question)
```

**Agent Process**:
1. Recognizes the need for factual information
2. Searches Wikipedia for "Tom M. Mitchell"
3. Extracts relevant information about his publications
4. Returns answer about "Machine Learning" book

## Python Agent

### Specialized Code Execution Agent

```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
```

### Data Processing Example

```python
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")
```

**Agent Execution**:
1. Understands the sorting requirement
2. Writes Python code to sort by last name, then first name
3. Executes the code using PythonREPLTool
4. Returns formatted sorted results

## Agent Types

### CHAT_ZERO_SHOT_REACT_DESCRIPTION

**Characteristics**:
- Zero-shot reasoning without prior examples
- Uses ReAct (Reason + Act) framework
- Optimized for chat models
- Performs reasoning step before taking action

**When to Use**:
- Immediate responses without training
- Dynamic tool selection based on context
- Multi-step reasoning tasks
- Interactive applications

**Note**: As of 2025, this agent type is deprecated. New applications should use newer agent constructors like `create_react_agent` or `create_structured_chat_agent`, or consider using LangGraph for more complex workflows.

## Custom Tool Definition

### Creating Custom Tools

```python
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

# Add custom tool to agent
agent = initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)
```

**Custom Tool Requirements**:
- Clear docstring describing functionality
- Proper type hints for inputs and outputs
- Focused, single-purpose functionality
- Error handling considerations

### Using Custom Tools

```python
try:
    result = agent("What's the date today?") 
except: 
    print("exception on external access")
```

## Agent Debugging and Monitoring

### Detailed Chain Inspection

```python
import langchain
langchain.debug = True

# Run with detailed logging
agent.run("Sort these customers by last name")

# Turn off debug mode
langchain.debug = False
```

**Debug Information Includes**:
- Tool selection reasoning
- Intermediate computation steps
- API calls and responses
- Error handling and recovery

## Advanced Agent Concepts

### Tool Selection Strategy

**Agent Decision Process**:
1. **Analysis**: Parse user input to understand intent
2. **Planning**: Determine which tools are needed
3. **Execution**: Call selected tools with appropriate parameters
4. **Synthesis**: Combine results into coherent response

### Error Handling

**Common Error Scenarios**:
- Tool unavailability or failure
- Invalid tool parameters
- Parsing errors in tool outputs
- Rate limiting and external service issues

**Error Handling Configuration**:
```python
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,  # Graceful error recovery
    verbose=True                 # Detailed error reporting
)
```

## Best Practices

### Tool Design

**Effective Tool Characteristics**:
- **Single Responsibility**: Each tool should have one clear purpose
- **Clear Documentation**: Comprehensive docstrings for LLM understanding
- **Robust Error Handling**: Graceful failure modes
- **Consistent Interface**: Standardized input/output formats

### Agent Configuration

**Performance Optimization**:
- Choose appropriate agent types for use cases
- Limit tool sets to reduce decision complexity
- Configure proper error handling
- Monitor token usage and costs

### Production Considerations

**Reliability Measures**:
- Implement fallback mechanisms for tool failures
- Set appropriate timeouts for external services
- Log agent decisions for debugging and improvement
- Test with diverse input scenarios

## Use Cases

### Data Analysis and Processing

**Applications**:
- Automated data cleaning and transformation
- Statistical analysis and reporting
- Visualization generation
- Database querying and manipulation

### Information Retrieval and Research

**Capabilities**:
- Multi-source information gathering
- Fact verification and cross-referencing
- Automated report generation
- Real-time data monitoring

### Task Automation

**Automation Scenarios**:
- Workflow orchestration
- System administration tasks
- Content generation and editing
- API integration and data synchronization

## Migration to Modern Frameworks

### Current Recommendations (2025)

**For New Projects**:
- **LangGraph**: More flexible and feature-rich agent framework
- **Modern Agent Constructors**: `create_react_agent`, `create_json_agent`
- **Tool-calling Models**: Native function calling capabilities
- **State Management**: Persistent conversation context

**Migration Benefits**:
- Enhanced debugging and observability
- Better state persistence
- Human-in-the-loop workflows
- More robust error handling
- Improved scalability and performance