# Question Answering

## Key LangChain Objects

- `langchain.chains.RetrievalQA`
- `langchain.indexes.VectorstoreIndexCreator`
- `langchain.document_loaders.CSVLoader`
- `langchain.vectorstores.DocArrayInMemorySearch`
- `langchain.chat_models.ChatOpenAI`
- `langchain.embeddings.OpenAIEmbeddings`

## Overview

Question Answering (QA) systems enable querying documents and datasets to retrieve specific information. Unlike simple search, QA systems understand context and generate natural language responses based on the retrieved content.

**Core Process**:
1. Load documents from various sources (CSV, PDF, web, etc.)
2. Create embeddings and store in vector database
3. Retrieve relevant documents based on query similarity
4. Generate answers using LLM with retrieved context

## Document Loading and Processing

**CSV Document Loading**:
```python
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()

# Each document contains page_content and metadata
print(docs[0])  # Shows first document structure
```

**Quick Setup with VectorstoreIndexCreator**:
```python
from langchain.indexes import VectorstoreIndexCreator

# Single-line vector store creation
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# Direct querying
query = "Please list all shirts with sun protection"
response = index.query(query, llm=llm)
```

## Step-by-Step QA Implementation

**1. Create Embeddings**:
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("sample text")
print(len(embed))  # Shows embedding dimension (1536 for OpenAI)
```

**2. Build Vector Database**:
```python
# Create vector store from documents
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Perform similarity search
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
print(len(docs))  # Default returns 4 similar documents
```

**3. Create Retriever**:
```python
retriever = db.as_retriever()
```

**4. Manual QA Process**:
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# Combine retrieved documents
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

# Generate response with context
response = llm.call_as_llm(f"{qdocs} Question: {query}")
```

## RetrievalQA Chain

**Automated QA Pipeline**:
```python
from langchain.chains import RetrievalQA

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",           # Stuff all documents into prompt
    retriever=retriever, 
    verbose=True                  # Show intermediate steps
)

query = "Please list all shirts with sun protection in a table"
response = qa_stuff.run(query)
```

## Chain Types

**Stuff Chain Type**:
- **How it works**: Combines all retrieved documents into a single prompt
- **Best for**: Small number of documents that fit within token limits
- **Advantages**: Single LLM call, simple and fast
- **Limitations**: Token limit constraints

**Other Chain Types**:
- **Map-Reduce**: Processes documents separately then combines results
- **Refine**: Iteratively refines answer by processing documents sequentially
- **Map-Rerank**: Scores each document's relevance and uses best ones

## Advanced Configuration

**Custom Embedding and Vector Store**:
```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,          # Use specific embedding model
).from_loaders([loader])
```

**RetrievalQA with Custom Parameters**:
```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"  # Custom document separator
    }
)
```

## Use Cases

**Product Catalogs**:
- Query product specifications and features
- Compare items across categories
- Find products matching specific criteria

**Document Libraries**:
- Research papers and technical documentation
- Policy documents and procedures
- Knowledge bases and FAQs

**Customer Support**:
- Automated response generation
- Information retrieval from manuals
- Ticket resolution assistance

## Best Practices

**Document Preparation**:
- Ensure clean, well-structured source data
- Include relevant metadata for filtering
- Consider document chunking for large files

**Query Optimization**:
- Use specific, well-formed questions
- Include context when necessary
- Test with various query phrasings

**Performance Considerations**:
- Monitor token usage with large document sets
- Consider chunking strategies for long documents
- Implement caching for frequently asked questions

**Model Selection**:
- Choose appropriate embedding models for domain
- Select LLM based on response quality needs
- Consider cost vs. performance trade-offs