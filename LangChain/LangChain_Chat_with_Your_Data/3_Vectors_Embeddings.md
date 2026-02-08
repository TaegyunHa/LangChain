# 3. Vectors and Embeddings

## Key LangChain Objects

- `langchain.embeddings.openai.OpenAIEmbeddings`
- `langchain.embeddings.openai.OpenAIEmbeddings.embed_query`
- `langchain.vectorstores.Chroma`
- `langchain.vectorstores.FAISS`

## Embeddings

> https://python.langchain.com/docs/concepts/embedding_models/

**What are Embeddings?**
Embeddings are vector representations of text that capture semantic meaning. They allow similarity search and retrieval based on meaning rather than exact keyword matches.

**Key Properties:**
- Texts with similar content will have similar vectors
- Can be compared using mathematical operations (cosine similarity, dot product, euclidean distance)
- Enable semantic search across unstructured data

**Basic embedding example**

```python
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
sentence1 = "i like dogs"
sentence2 = "i like canines"
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
# Higher result means two embeddings are more similar
np.dot(embedding1, embedding2)
```

## Vector Stores

**What are Vector Stores?**
Vector stores are specialized databases that store embedding vectors of document chunks. They enable efficient similarity search and retrieval of semantically related content.

**How Vector Stores Work:**
1. Documents are split into chunks
2. Each chunk is converted to an embedding vector
3. Vectors are stored in the vector store with metadata
4. Queries are embedded and compared against stored vectors
5. Most similar chunks are retrieved based on vector similarity

**Popular Vector Store Options:**
- **Local:** Chroma, FAISS
- **Cloud:** Pinecone, MongoDB Atlas, Qdrant, PostgreSQL (PGVector)

**Vector store example**
```python
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3)
docs[0].page_content
vectordb.persist()
```
**Key Methods and Parameters:**
- `vectordb._collection.count()` - Returns number of stored documents (same as `len(splits)`)
- `vectordb.similarity_search(query, k=n)` - Returns n most similar documents
- `vectordb.persist()` - Saves vector store to disk
- `k` parameter - Controls number of documents returned
- `filter` parameter - Allows filtering by metadata

## RAG (Retrieval-Augmented Generation)

**What is RAG?**
RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language model generation to provide accurate, contextual responses using external knowledge.

**How RAG Works in LangChain:**
1. **Retrieval:** Find relevant context from vector store based on user query
2. **Augmentation:** Add retrieved information to the LLM prompt  
3. **Generation:** Generate response using both query and retrieved context

**Key RAG Components:**
- **Vector Stores:** Store and search document embeddings (e.g., Chroma)
- **Retrievers:** Find relevant documents using semantic similarity
- **LLMs:** Generate answers using retrieved context
- **Document Loaders:** Load and process source documents

**Benefits:**
- Enables LLMs to answer questions about specific documents
- Provides more accurate and grounded responses
- Works with knowledge not in the model's training data
- Reduces hallucination by grounding responses in actual documents

## Advanced Retrieval Techniques

**Similarity Search Types:**
- **Basic Similarity:** Standard vector similarity search
- **MMR (Maximum Marginal Relevance):** Diversifies results to reduce redundancy
- **Score Thresholding:** Filters results below similarity threshold
- **Top-K:** Returns specified number of most similar documents

**Performance Considerations:**
- Modern vector stores can handle billions of embeddings
- Sub-50ms query latency possible at scale
- HNSW (Hierarchical Navigable Small World) algorithm commonly used for efficient search