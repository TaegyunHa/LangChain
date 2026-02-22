# RAG and Agents

## Overview

**Context construction** is the process of providing a model with the necessary information to solve a task. While instructions are common to all queries, the context is specific to each query.

This chapter covers:
- **RAG (Retrieval-Augmented Generation)** retrieves relevant information from external sources to improve model responses, reducing hallucination and token usage
- **Retrieval algorithms** from term-based (BM25, Elasticsearch) to embedding-based (vector search, ANN), with hybrid approaches combining both
- **Agents** as autonomous entities that use tools to perceive their environment and take actions, with RAG as a specialised agent pattern
- **Planning and memory** systems that enable agents to decompose tasks, reflect on outcomes, and persist information across sessions

---

## 1. RAG - Retrieval-Augmented Generation

### 1.1 Why RAG Matters

> **Key Insight**: Longer context doesn't mean better use of context — the longer the context, the more likely the model is to focus on the wrong part.

RAG improves a model's generation by retrieving the relevant information from external sources. It serves as context construction — analogous to feature engineering in traditional ML.

| Aspect | Without RAG | With RAG |
|--------|------------|----------|
| **Context** | Entire knowledge base or none | Only relevant chunks |
| **Token usage** | High (long context) | Reduced (targeted retrieval) |
| **Performance** | May focus on wrong context | Improved relevance |
| **Knowledge** | Limited to training data | Extended via external sources |

External sources can include:
- Internal databases
- Previous chat sessions
- The internet

> **Key Takeaway**: Regardless of a model's context capacity, RAG remains important. It both reduces input tokens and increases model performance.

### 1.2 RAG Architecture

A RAG system has two core components:
- **Retriever**: Fetches relevant information from external sources (indexing + querying)
- **Generator**: Produces a response based on retrieved information

![RAG-Architecture](./images/RAG-arch.png)

The two components are often trained separately, but fine-tuning the whole RAG system end-to-end can improve performance significantly.

**The quality of the retriever defines the success of RAG.**

The retriever has two functions:
- **Indexing**: Processing data to enable efficient retrieval later
- **Querying**: Retrieving data chunks most relevant to the query

**Chunking strategy**: To avoid arbitrary context sizes, split each document into more manageable chunks. The goal is to retrieve the data chunk most relevant to the query.

---

## 2. Retrieval Algorithms

### 2.1 Term-Based Retrieval

> **Key Insight**: Term-based retrieval works well out of the box and is faster during both indexing and querying.

Lexical (keyword-based) retrieval ranks documents by term overlap with the query.

#### Core Concepts

| Problem | Solution | How It Works |
|---------|----------|-------------|
| Insufficient context space | **Term Frequency (TF)** | The more a term appears, the more relevant the document |
| Some terms matter more than others | **Inverse Document Frequency (IDF)** | Term importance is inversely proportional to the number of documents containing it |
| Need combined relevance | **TF-IDF** | Combines TF and IDF for balanced scoring |

- **IDF Formula**: `IDF = nTotalDocuments / nDocumentsWithTerm`
  - Higher IDF = more important term
- **Common implementations**:
  - Elasticsearch, BM25

#### Tokenisation for Retrieval

Tokenisation is the process of breaking a query into individual terms:
- Split the query into words (risk: loses meaning, e.g. "hot dog" → "hot", "dog")
- Convert all characters to lowercase
- Remove punctuation
- Eliminate stop words (e.g. "the", "and", "is")
- Use n-grams to preserve multi-word meaning

**Limitation**: N-gram overlap between query and document works best when lengths are similar. It becomes difficult to distinguish relevance when documents are much longer than the query.

### 2.2 Embedding-Based Retrieval

> **Key Insight**: Embedding-based retrieval measures how closely the *meaning* of a document aligns with the query, not just keyword overlap.

#### Indexing and Querying

- **Indexing**: Convert original data chunks into embeddings → stored in a vector database
- **Querying**:
  1. Convert query into an embedding
  2. Fetch k chunks with closest embeddings
  3. Rerank candidates

**Important considerations**:
- Embedding-based retrieval doesn't work if the embedding model is poor
- Caching can reduce latency
- In vector databases, storing is easy — **vector search is hard**
- Vectors must be indexed and stored for fast, efficient search

#### Vector Search Algorithms

Vector search is fundamentally a **nearest neighbour search problem**.

| Algorithm | How It Works | Trade-off |
|-----------|-------------|-----------|
| **KNN (K-Nearest Neighbours)** | Compute similarity scores between query and all vectors using cosine similarity, rank, return top k | Precise but computationally heavy — only for small datasets |
| **ANN (Approximate Nearest Neighbour)** | Organise vectors into buckets, trees, or graphs; use quantised/sparse vectors | Faster, scalable, but approximate |

#### ANN Algorithms

| Algorithm | Approach |
|-----------|----------|
| **LSH** (Locally Sensitive Hashing) | Hashing to speed up similarity search |
| **HNSW** (Hierarchical Navigable Small World) | Multi-layer graph where nodes represent vectors; nearest neighbour search by traversing graph edges |
| **Product Quantisation** | Lower-dimensional representation by decomposing each vector into multiple subvectors |
| **IVF** (Inverted File Index) | Organise similar vectors into the same cluster |
| **Annoy** (Approximate Nearest Neighbors Oh Yeah) | Multiple binary trees |

### 2.3 Comparing Retrieval Algorithms

| Aspect | **Term-Based** | **Embedding-Based** |
|--------|---------------|-------------------|
| **Indexing speed** | Faster | Slower |
| **Query speed** | Faster | Slower |
| **Computational cost** | Less expensive | More expensive |
| **Out-of-the-box quality** | Works well | Requires good embedding model |
| **Improvement potential** | Limited | Significant (fine-tune embeddings) |
| **Keyword handling** | Strong | Can obscure keywords |

### 2.4 Combining Retrieval Algorithms

**Hybrid search** combines term-based and embedding-based retrieval via reranking:

```
Stage 1: Cheap, less precise term-based retriever → fetches candidates
Stage 2: More precise, more expensive embedding-based retriever → finds the best candidates
```

---

## 3. Retrieval Evaluation and Optimisation

### 3.1 Evaluation Metrics

- **Context Precision**: Of all documents retrieved, what percentage is relevant?
- **Context Recall**: Of all relevant documents, what percentage was retrieved?

Some RAG frameworks only support context precision as it is simpler to compute.

#### ANN Benchmarks

| Metric | What It Measures |
|--------|-----------------|
| **Recall** | Fraction of nearest neighbours found by the algorithm |
| **QPS** (Queries Per Second) | Number of queries handled per second |
| **Build time** | Time required to build the index |
| **Index size** | Size of the index |

> **Key Takeaway**: Quality of a RAG system should be evaluated both component-by-component and end-to-end: (1) retrieval quality, (2) final RAG output, (3) embedding quality.

### 3.2 Retrieval Optimisation Techniques

Four primary techniques to improve retrieval results:

#### Chunking Strategy

| Approach | Description |
|----------|------------|
| **Fixed-size chunking** | e.g. 512 tokens per chunk |
| **Sentence/paragraph-based** | Split at natural boundaries |
| **Recursive splitting** | Hierarchy: document → section → paragraph → sentence |
| **Semantic chunking** | Split based on topic or meaning boundaries |

**Trade-off**: Smaller chunks enable more precise retrieval but may lose surrounding context. Larger chunks preserve context but may include irrelevant information.

**Overlap** between chunks can help preserve context at boundaries.

#### Reranking

A two-stage retrieval pipeline:

| Stage | Role | Characteristics |
|-------|------|----------------|
| **Initial retrieval** | Fetch broad set of candidates (~50 documents) | Cheap, less precise |
| **Reranking** | Score and reorder candidates | More precise, more expensive |

Rerankers can be:
- **Cross-encoder models** — jointly encode query and document for more accurate relevance scoring
- **LLM-based rerankers**

Reranking improves precision without running the expensive model over the entire corpus.

#### Query Rewriting

- Rewrite the query to reflect what the user is actually asking
- The new query should make sense on its own, without prior conversation context
- Example: "How about her?" → "How about Aunt Mabel from the previous question?"
- Prompt pattern: *"Given the following conversation, rewrite the last user input to reflect what the user is actually asking"*
- The rewriting model should acknowledge when a query is not solvable rather than hallucinating
- **Trade-off**: Adds latency due to the additional generation step

#### Contextual Retrieval

- "Chunks-for-chunks" approach: retrieve supplementary context for initially retrieved chunks
- After fetching initial chunks, retrieve additional metadata, tags, or related chunks to enrich context
- Can include:
  - Document-level metadata (title, author, date)
  - Section headers or summaries
  - Neighbouring chunks from the same document

### 3.3 Cost Considerations

- Vector database expenses can consume between **one-fifth to one-half** of total model API spending
- Includes both storage and query costs
- Data changes frequently → frequent embedding regeneration required
- More detailed the index, more accurate the retrieval, but slower

---

## 4. RAG Beyond Text

### 4.1 Multimodal RAG

Extend RAG beyond text to include images, audio, and video.

| Approach | How It Works |
|----------|-------------|
| **Modality conversion** | Convert non-text to text (image captioning, speech-to-text), then use standard text RAG |
| **Multimodal embeddings** | Embed different modalities into a shared vector space |
| **Multimodal LLMs** | Directly process mixed-modality retrieved content |

**Challenges**: Aligning embeddings across modalities; evaluating retrieval quality for non-text data.

### 4.2 RAG with Tabular Data

Structured data (tables, databases) requires different retrieval strategies.

| Approach | How It Works |
|----------|-------------|
| **Text-to-SQL** | Convert natural language query into SQL to retrieve from databases |
| **Table serialisation** | Convert tables to text format for embedding and retrieval |
| **Hybrid** | Combine structured queries with semantic search over table descriptions |

**Challenges**: Preserving table structure and relationships; handling large tables that exceed context limits.

---

## 5. Agents

### 5.1 What Is an Agent?

> **Key Insight**: An agent is "an entity capable of perceiving its environment and acting upon it."

An agent is characterised by:
- **The environment** it operates in
- **The actions/tools** available to it

Agents use tools to gather information and take actions to accomplish tasks. **RAG can be viewed as a specialised agent pattern** where the retriever functions as a tool.

### 5.2 Tools

Tools extend what an agent can do beyond the base model's capabilities.

| Category | Purpose | Examples | Risk Level |
|----------|---------|----------|-----------|
| **Knowledge Augmentation** | Provide information not in training data | RAG systems, web search, API calls (weather, stocks, news) | Low |
| **Capability Extension** | Perform actions the model cannot do alone | Code interpreters, terminal access, function execution | Medium |
| **Write Actions** | Change state in the real world | Database CRUD, file operations, send emails, post messages | High |

Capability extension tools significantly boost performance compared to prompting or fine-tuning alone. Write actions are the most impactful but also the **riskiest** — they change state in the real world.

### 5.3 Planning

#### Foundation Models as Planners

- There is debate about whether autoregressive models can truly plan
- In practice, foundation models can decompose tasks and generate step-by-step plans
- Planning quality depends heavily on model capabilities and prompt design

#### Plan Generation

![plan-exe](./images/plan-exe.png)
_Decoupling planning and execution to execute only validated plans_

**Four-stage planning cycle**:

| Stage | Purpose |
|-------|---------|
| **1. Plan generation** | Decompose the task into sub-tasks |
| **2. Initial reflection** | Evaluate the plan before execution |
| **3. Execution** | Carry out the plan via function calls and tool use |
| **4. Final reflection** | Evaluate the outcome and adjust if needed |

**Ways to improve plan generation**:
- Enhance system prompts with examples
- Provide better tool descriptions and documentation
- Refactor complex functions into simpler ones
- Use stronger models or fine-tune for planning

#### Function Calling

The mechanism by which agents invoke tools. Requires:
- **Tool inventory**: Function names, parameters, and documentation
- **Usage specifications**: Required vs optional parameters

The model generates structured output (e.g. JSON) specifying which function to call and with what arguments.

#### Planning Granularity

> **Key Insight**: Higher-level plans are easier to generate but harder to execute, while detailed plans are harder to generate but easier to execute.

| Approach | Advantage | Disadvantage |
|----------|-----------|-------------|
| **High-level plans** | Easier to generate | Harder to execute |
| **Detailed plans** | Easier to execute | Harder to generate |
| **Hierarchical** | Generate high-level first, then refine to finer level | Best of both |

**Tool inventory and reusability**:
- Using exact function names makes it harder to reuse a planner across different use cases (fine-tuning required)
- Plans can instead be generated using natural language
- This requires an additional "translator" to convert each natural language action into an executable command — a simpler task than planning with lower hallucination risk

#### Complex Plan Execution Patterns

| Pattern | Description | Trade-off |
|---------|------------|-----------|
| **Sequential** | Actions performed one after another | Predictable but potentially slow |
| **Parallel** | Concurrent actions when no dependencies | Faster but adds complexity |
| **Conditional** | Decision points (if-statements) adapting to intermediate results | Flexible but harder to debug |
| **Iterative** | Loops for repetitive tasks over datasets | Efficient but risk of infinite loops |

#### Reflection and Error Correction

- Agents should evaluate their own outputs and actions
- When an action fails or produces unexpected results:
  - Reflect on what went wrong
  - Adjust the plan or try alternative approaches
  - Avoid repeating the same failed action
- Self-reflection can be built into the agent loop as an explicit step

### 5.4 Tool Selection

> **Key Insight**: More tools give more capabilities; however, the more tools there are, the harder it is to efficiently use them.

**Systematic approach to tool evaluation**:
- Conduct ablation studies to measure each tool's impact on performance
- Monitor usage patterns and error rates
- Analyse call distribution across tools

**Model-specific preferences**: Different models may prefer different tool sets. GPT-4 tends to use broader tool sets than ChatGPT.

---

## 6. Agent Failure Modes and Evaluation

### 6.1 Planning Failures

The most common mode of planning failure is **tool use failure**.

| Failure Type | Description | Example |
|-------------|------------|---------|
| **Invalid tool** | Model selects a tool that does not exist | Calling `search_database` when only `query_db` exists |
| **Valid tool, invalid parameters** | Correct tool but parameters don't match schema | Missing required field, wrong data type |
| **Valid tool, incorrect values** | Correct tool and parameters, but values are wrong | Passing wrong date format, nonsensical input |

**Mitigation**: Clear tool documentation, constrained tool selection, schema validation, few-shot examples.

### 6.2 Tool Failures

The tool itself executes but produces wrong or unusable outputs:
- External service downtime or rate limiting
- Stale or incorrect data returned by the tool
- Misunderstanding of tool capabilities leading to misinterpreted results

**Mitigation**: Robust error handling, fallback strategies, output validation, retry logic.

### 6.3 Efficiency Evaluation

| Metric | What It Measures |
|--------|-----------------|
| **Average steps** | Number of steps the agent needs to complete a task |
| **Cost per task** | Token usage, API calls, tool invocations |
| **Latency** | Time taken per action in the pipeline |

---

## 7. Memory

### 7.1 Memory Types

Memory allows a model to retain and utilise information across interactions.

| Memory Type | Description | Persistence | Example |
|------------|------------|-------------|---------|
| **Internal knowledge** | What the model learnt during training (parametric memory) | Permanent but static | General world knowledge |
| **Short-term memory** | The current conversation context (context window) | Session only | Current chat history |
| **Long-term memory** | Persistent storage across sessions | Cross-session | External databases, vector stores |

### 7.2 Why Memory Matters

Memory is essential for:
- Managing information overflow within limited context windows
- Maintaining consistency across interactions
- Persisting user preferences and prior decisions
- Preserving structural integrity of complex data

---

## Summary

- RAG and agents are both **prompt-based methods** — they influence the model's quality solely through inputs without modifying the model itself
- **RAG** was originally developed to overcome a model's context limitations, but also enables more efficient use of information, improving response quality while reducing costs
  - Two-step process: retrieve relevant information from external memory, then generate more accurate responses
  - The success of a RAG system depends on the quality of its retriever
  - Term-based retrievers (Elasticsearch, BM25): lighter to implement, strong baselines
  - Embedding-based retrievers: more computationally intensive, but potential to outperform term-based algorithms
  - Embedding-based retrieval is powered by **vector search**, also the backbone of core internet applications (search, recommender systems)
  - Optimisation through chunking, reranking, query rewriting, and contextual retrieval
  - Evaluation via context precision and recall
- **RAG is a special case of an agent** where the retriever is a tool the model can use
- **Agents** are defined by their environment and the tools they can access
  - AI is the planner that analyses its given task, considers different solutions, and picks the most promising one
  - Planning can be augmented with **reflection** and a **memory system** to help keep track of progress
  - More tools → more capabilities → more challenging tasks solvable
  - However, **the more automated the agent becomes, the more catastrophic its failures**
  - Tool use exposes agents to many security risks; rigorous defensive mechanisms need to be in place
- Both RAG and agents work with a lot of information, often exceeding the maximum context length → necessitates a **memory system** for managing and using all the information
- While RAG and agents can enable many incredible applications, modifying the underlying model can open up even more possibilities (next chapter: fine-tuning)

---

## Glossary of Key Terms

| **Term** | **Definition** |
|----------|---------------|
| **ANN** | Approximate Nearest Neighbour; scalable vector search trading precision for speed |
| **BM25** | Best Matching 25; probabilistic term-based retrieval algorithm |
| **Chunking** | Splitting documents into smaller, manageable pieces for retrieval |
| **Context Precision** | Of all retrieved documents, the percentage relevant to the query |
| **Context Recall** | Of all relevant documents, the percentage successfully retrieved |
| **Cross-Encoder** | Model that jointly encodes query and document for accurate relevance scoring |
| **Embedding** | Vector representation capturing semantic meaning of text |
| **HNSW** | Hierarchical Navigable Small World; graph-based ANN algorithm |
| **Hybrid Search** | Combining term-based and embedding-based retrieval |
| **IDF** | Inverse Document Frequency; measures term importance across documents |
| **IVF** | Inverted File Index; clusters similar vectors for efficient search |
| **KNN** | K-Nearest Neighbours; exact nearest neighbour search |
| **LSH** | Locally Sensitive Hashing; hash-based approximate search |
| **Product Quantisation** | Compresses vectors by decomposing into subvectors |
| **RAG** | Retrieval-Augmented Generation; combining retrieval with generation |
| **Reranking** | Two-stage retrieval: broad fetch followed by precise scoring |
| **TF-IDF** | Term Frequency-Inverse Document Frequency; combined relevance metric |
| **Vector Database** | Database storing embeddings for similarity search |

---

## Key Takeaways

> **RAG**
> - Retrieves relevant information from external sources to improve generation quality
> - Reduces token usage and improves performance regardless of context window size
> - Success depends on retriever quality — evaluate both component-by-component and end-to-end
> - Optimise through chunking, reranking, query rewriting, and contextual retrieval

> **Retrieval Algorithms**
> - Term-based (BM25, Elasticsearch): fast, cheap, works well out of the box
> - Embedding-based: more expensive but can be significantly improved over time
> - Hybrid approach combines the strengths of both via reranking

> **Agents**
> - Defined by their environment and available tools
> - Use planning (with reflection) to decompose and execute tasks
> - More tools → more capabilities → more challenging tasks solvable
> - **The more automated the agent, the more catastrophic its failures**

> **Memory**
> - Three types: internal knowledge, short-term (context window), long-term (persistent storage)
> - Essential for managing information overflow and maintaining consistency
> - Both RAG and agents often exceed context limits, necessitating memory systems

> **RAG and Agents Relationship**
> - Both are prompt-based methods — they influence quality through inputs without modifying the model
> - RAG is a special case of an agent where the retriever is a tool
> - While powerful, modifying the underlying model (fine-tuning) can unlock further possibilities
