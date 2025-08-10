# Topics

- Models
	- LLMs
	- Chat Models
	- Text Embedding Models
- Prompts
	- Prompt Templates
	- Output Parsers
		- Retry/fixing
	- Example Selectors
- Indexes
	- Document Loaders
	- Text Splitters
	- Vector Stores
	- Retrievers
- Chains
	- Prompt + LLM + Output Parsing
- Agents
	- Algorithms for getting LLMs to use tools

# LangChain Objects

- `langchain.chat_models.ChatOpenAI`
- `langchain.prompts.ChatPromptTemplate`
- `langchain.output_parsers.ResponseSchema`
- `langchain.output_parsers.StructuredOutputParser`

## ChatOpenAI

> https://python.langchain.com/docs/concepts/chat_models/

The `ChatOpenAI` object helps make prompts reproducible in a controlled manner when used with `ChatPromptTemplate`.
It can be thought of as a model adapter to use with `ChatPromptTemplate`.

The object takes a **list of messages** as input and returns a **message** as output.

With direct API calls, prompting needs to be manually crafted and passed into `messages`:

```python
import openai
response = openai.ChatCompletion.create(
	model="gpt-3.5-turbo",
	messages="my prompt goes here",
	temperature=0)
```

With `ChatOpenAI`, we can define the model first, then use `ChatPromptTemplate` later for fine and reproducible prompting control:

```python
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(
	model="gpt-3.5-turbo",
	temperature=0.0)
```

### Questions

- Why do you put `\` at the end of each line of the prompt template?
	- `\` is a line continuation character in Python. It removes `\n` and makes a string into one line.
- Why set temperature to 0?
	- Randomness and creativity of generated text can be controlled by temperature.
	- A higher value (e.g., 1.0) makes responses more creative
	- A lower value (e.g., 0.0) makes them more deterministic and focused

## ChatPromptTemplate

> https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

Translates user input and parameters into instructions for a model.
`ChatPromptTemplate` will be used to format a list of messages.

- **Input**: dict
	- **key**: a variable in the prompt template to fill in
	- **value**: a value replacing a variable in the prompt
- **Output**: `PromptValue`
	- can be passed into `ChatModel`
	- can be cast to a string or list of messages
	- `PromptValue` allows switching between string and message formats

**Direct API Example**

```python
import openai
email = """Hello there!"""
style = """British English"""
prompt = f"""Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{email}```"""
response = openai.ChatCompletion.create(
	model="gpt-3.5-turbo",
	messages=prompt,
	temperature=0)
```

**Prompt Template Example 1**

```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt_template.invoke({"topic": "cats"})
```

- Above template consists of:
	1. system message without variables to format
	2. human message that can be formatted by `topic`

**Prompt Template Example 2**

```python
from langchain.prompts import ChatPromptTemplate

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
# Make prompt template
prompt_template = ChatPromptTemplate.from_template(template_string)
# Access prompt
prompt_template.messages[0].prompt
# Access prompt input vars
prompt_template.messages[0].prompt.input_variables
# Create a prompt from template
customer_style = "British English"
customer_email = "Ah, I be fuming that my blender lid flew off!"
customer_messages = prompt_template.format_messages(
	style=customer_style,
	text=customer_email)
type(customer_messages[0]) # <class 'langchain.schema.HumanMessage'>

# Get response from the chat model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
customer_response = chat(customer_messages)
```

## StructuredOutputParser

It's possible to get JSON-formatted **strings** from a chat model by adding **format instructions** in the prompt.
However, as it's a `str` not a `dict`, additional parsing is needed. This process can be improved by using `StructuredOutputParser`.

With output parser:
- format instructions can be created
- output responses can be parsed into `dict`

### ResponseSchema

`ResponseSchema` is a building block of `StructuredOutputParser` that defines format instructions and the structure of parsed output `dict`.

**Output format of get_format_instruction**

```md
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\`\`\`json" and "\`\`\`":

\`\`\`json
{
	"<ResponseSchema.name>": string // <ResponseSchema.description>
}
\`\`\`
```

- Why `:string`? 
	- This indicates the expected data type for the field.

**Example of StructuredOutputParser**

```python
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased as a gift for someone else? \
                             Answer True if yes, False if not or unknown.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any sentences about the value or \
                                    price, and output them as a comma separated Python list.")
response_schemas = [gift_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instruction = output_parser.get_format_instruction()

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template)
messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instruction)
response = chat(messages)
output_dict = output_parser.parse(response.content)
```

### Thoughts

- I don't like the fact that there is duplication between `ResponseSchema` and template.
	- I tried to avoid duplication by following the template, but it failed to get the correct value for `gift`.

```python
"""
For the following text, extract the information using format_instruction:

text: {text}

format_instruction: {format_instructions}
"""
```
