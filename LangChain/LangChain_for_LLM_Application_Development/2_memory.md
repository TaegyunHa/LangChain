# Memory

LLMs are stateless
- Each transaction is independent
- Memory can be achieved by providing the full conversation as context
- If we keep sending the full conversation, it becomes large fast; hence, expensive.

Memory objects in LangChain handle the following:
- Managing conversation history
    - Keep only the last n turns of the conversation between the user and the AI.
- Extraction of structured information
    - Extract structured information from the conversation history, such as a list of facts learned about the user.
- Composite memory implementations
    - Combine multiple memory sources,
    - i.e. a list of known facts about the user along with facts learned during a given conversation.

We use the `ConversationChain` instead of `ChatPromptTemplate` to implement memory.

# LangChain Objects

Note that all of the following are deprecated in favor of `LangGraph`:

- `langchain.chains.ConversationChain`
- `langchain.memory.ConversationBufferMemory`
- `langchain.memory.ConversationBufferWindowMemory`
- `langchain.memory.ConversationTokenBufferMemory`
- `langchain.memory.ConversationSummaryMemory`

# Memory Types in LangChain

LangChain offers various memory types to handle memory.
- Multiple memories can be stored at one time.
    - i.e. Conversation memory + Entity memory for an individual
- Memory can be stored in a conventional database (like SQL)
- Types
    - ConversationBufferMemory
        - Stores messages in history
        - Extracts the messages in a variable
    - ConversationBufferWindowMemory
        - Stores messages in history
        - Limits the number of interactions stored in memory (`k`)
    - ConversationTokenBufferMemory
        - Stores messages in history
        - Limits the token length of stored interactions in memory
    - ConversationSummaryMemory
        - Creates a summary of the conversation over time
        - Limits the token length of stored summarized memory
    - Vector data memory
        - Stores messages in a vector database and retrieves the most relevant block of messages
    - Entity memory
        - Stores details about specific entities

## ConversationBufferMemory

Instead of passing a prompt directly into `ChatOpenAI`, we can achieve memory through the `ConversationChain` with `ConversationBufferMemory`.

**Example with memory**

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)
conversation.predict(input="Hi, my name is Andrew")
```

**Example without memory**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
prompt_template = ChatPromptTemplate.from_template(template_string)
customer_messages = prompt_template.format_messages(
	style="British English",
	text="Hi, I'm Taegyun")
llm(customer_messages)
```

