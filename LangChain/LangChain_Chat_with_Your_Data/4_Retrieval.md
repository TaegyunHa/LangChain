# Retrieval

When a query comes in, the system needs to retrieve the most relevant document splits from the vector store. This is where retrieval techniques become crucial for building effective.

## Core Concepts

**Retriever Interface**: A lightweight wrapper around vector stores that standardizes document retrieval. All retrievers implement the same interface with a `get_relevant_documents()` method.

**Basic Retriever Creation**:
```python
retriever = vectorstore.as_retriever()
```

## Key LangChain Objects

- `langchain.vectorstores.Chroma.max_marginal_relevance_search`
- `langchain.retrievers.contextual_compression.ContextualCompressionRetriever`
- `langchain.retrievers.self_query.base.SelfQueryRetriever`

## Search Types and Parameters

**Common Search Types**:
- `similarity` (default): Standard semantic similarity search
- `mmr`: Maximum Marginal Relevance for diverse results
- `similarity_score_threshold`: Filter by minimum relevance score

**Key Parameters**:
- `k`: Number of documents to return (default: 4)
- `fetch_k`: Documents to fetch before MMR filtering (default: 20) 
- `lambda_mult`: MMR diversity parameter (0=max diversity, 1=min diversity, default: 0.5)
- `score_threshold`: Minimum relevance threshold for filtering

## Maximum Marginal Relevance (MMR)

Most similar responses may not always be preferred. A more diverse set of splits can be more beneficial, especially when dealing with redundant or overlapping content.

**Purpose**: MMR balances relevance and diversity to avoid retrieving multiple similar documents that provide redundant information.

**MMR Algorithm**:
1. Query the vector store using semantic similarity
2. Choose the top `fetch_k` most similar responses  
3. From those responses, iteratively select the `k` most diverse documents
4. Each selection considers both similarity to query and dissimilarity to already selected documents

**When to Use MMR**:
- Large document collections with potential redundancy
- Applications requiring diverse perspectives on a topic
- Maximizing information coverage within limited context windows

**Example - Direct MMR Search**:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]
smalldb = Chroma.from_texts(texts, embedding=embedding)

# Direct MMR search on vectorstore
docs = smalldb.max_marginal_relevance_search(
    query="Tell me about all-white mushrooms with large fruiting bodies",
    k=2,        # Return 2 diverse documents
    fetch_k=3   # Consider all 3 documents for diversity selection
)
```

**Example - MMR via Retriever**:

```python
# Using retriever interface with MMR
retriever = smalldb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 3,
        "lambda_mult": 0.5  # Balance between relevance and diversity
    }
)
docs = retriever.get_relevant_documents("Tell me about all-white mushrooms with large fruiting bodies")
```

## Self-Query Retrieval (LLM-Aided Retrieval)

Traditional retrieval only uses semantic similarity. However, queries often contain both semantic content and metadata criteria. Self-query retrievers use LLMs to parse natural language queries into structured queries with both semantic and metadata filters.

**Purpose**: Enable retrieval based on both document content and metadata attributes (source, date, author, etc.).

**How Self-Query Works**:
1. **User Question**: Natural language query containing both content and metadata references
2. **LLM Query Parser**: Extracts semantic query and metadata filters  
3. **Structured Query**: Combines similarity search with metadata filtering
4. **Filtered Results**: Returns documents matching both content relevance and metadata criteria

**Example Query**: "What did they say about regression in the third lecture?"
- **Semantic part**: "regression" 
- **Metadata filter**: `source == "Lecture03.pdf"`

**Traditional Approach (Manual Filtering)**:
```python
vectordb = Chroma(
    persist_directory='docs/chroma/',
    embedding_function=embedding
)
# You must manually specify the filter
docs = vectordb.similarity_search(
    "regression concepts",  # Manual extraction of semantic part
    k=3,
    filter={"source": "docs/cs229_lectures/MachineLearning-Lecture03.pdf"}  # Manual filter
)
```

**Self-Query Approach (Automatic Parsing)**:

```python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define metadata schema for the LLM to understand
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

# Create self-query retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# Natural language query - LLM will parse this automatically
question = "what did they say about regression in the third lecture?"
docs = retriever.get_relevant_documents(question)

# LLM automatically extracts:
# - Query: "regression" 
# - Filter: source contains "Lecture03"
for d in docs:
    print(d.metadata)
```

**Benefits of Self-Query**:
- Automatic parsing of complex queries
- No manual filter construction required  
- More natural language query interface
- Supports complex metadata combinations

## Contextual Compression

Traditional retrieval returns entire document chunks, which may contain irrelevant information. Contextual compression addresses this by filtering and compressing retrieved documents to include only query-relevant content.

**Purpose**: Improve retrieval precision by eliminating irrelevant information while preserving relevant context.

**Two-Phase Process**:
1. **Retrieval Phase**: Fetch potentially relevant documents (optimize for recall)
2. **Compression Phase**: Extract only relevant segments from retrieved documents (optimize for precision)

**Contextual Compression Workflow**:
1. Base retriever finds relevant document splits
2. Retrieved documents are passed to a document compressor
3. Compressor extracts only relevant segments using query context
4. Compressed, focused content is returned for downstream processing

**Benefits**:
- Reduces noise in retrieved content
- Enables fetching more documents initially (higher recall) while maintaining focused results
- More efficient use of context windows in language models
- Better relevance in final responses

**Available Compressors**:

1. **LLMChainExtractor**: Uses LLM to extract relevant segments from each document
2. **EmbeddingsFilter**: Filters documents by embedding similarity (faster, cheaper)
3. **DocumentCompressorPipeline**: Combines multiple compression techniques

**Example - Basic Compression**:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI

# Set up LLM-based compressor
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever with standard similarity search
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
# Returns only relevant segments, not full documents
```

**Example - Compression + MMR**:

```python
# Combine contextual compression with MMR for diverse, relevant results
compression_retriever_mmr = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever_mmr.get_relevant_documents(question)
# Returns diverse, compressed segments
```

**Example - Embeddings Filter (Faster Alternative)**:

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

# Cheaper, faster compression using embeddings similarity
embeddings_filter = EmbeddingsFilter(
    embeddings=embedding,
    similarity_threshold=0.76  # Only keep docs above this similarity
)

compression_retriever_filter = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectordb.as_retriever()
)
```

## Retrieval Strategy Selection Guide

**Use Standard Similarity Search When**:
- Simple semantic queries without metadata requirements
- Fast response times are critical
- Document content is already well-segmented

**Use MMR When**:
- Documents contain redundant or similar information
- Applications require diverse perspectives on a topic
- Working with large document collections
- Context window limitations require diverse information

**Use Self-Query When**:
- Queries reference specific metadata (dates, sources, authors)
- Users ask questions in natural language with implicit filters
- Complex metadata-based filtering is needed
- Manual filter construction should be avoided

**Use Contextual Compression When**:
- Retrieved documents contain significant irrelevant content
- Context windows are limited and precision is critical
- Applications need to maximize information density
- Working with lengthy documents or academic papers

**Combining Strategies**:
Most effective retrieval systems combine multiple techniques:
- MMR + Compression: Diverse, relevant segments
- Self-Query + Compression: Metadata-filtered, compressed results
- All three: Maximum precision and flexibility