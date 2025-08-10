# Document Spliting

> https://python.langchain.com/docs/concepts/text_splitters/#approaches

Document splitting is a crucial preprocessing step. It involves breaking down large texts into smaller, manageable and meaningful chunks.

This process offers several benefits:
- ensuring consistent processing of varying document lengths
- overcoming input size limitations of models
- improving the quality of text representations used in retrieval systems

# LangChain Objects

- `langchain.text_splitter.RecursiveCharacterTextSplitter`
- `langchain.text_splitter.CharacterTextSplitter`
- `langchain.text_splitter.TokenTextSplitter`
- `langchain.text_splitter.MarkdownHeaderTextSplitter`

## Goal of Document Splitting

- Handling non-uniform document lengths:
    - Real-world document collections often contain texts of varying sizes.
    - Splitting ensures consistent processing across all documents.
- Overcoming model limitations:
    - Many embedding models and language models have maximum input size constraints.
    - Splitting allows us to process documents that would otherwise exceed these limits.
- Improving representation quality: 
    - For longer documents, the quality of embeddings or other representations may degrade as they try to capture too much information.
    - Splitting can lead to more focused and accurate representations of each section.
- Enhancing retrieval precision:
    - In information retrieval systems, splitting can improve the granularity of search results, allowing for more precise matching of queries to relevant document sections.
- Optimizing computational resources: 
    - Working with smaller chunks of text can be more memory-efficient and allow for better parallelization of processing tasks.

# Strategy

Different splitting method can be used for each usecase:
- Length based
- Text structured based
- Semantic meaning based

## Length-based

Length based splitting ensures that each chunk doesn't exceed a specified size limit. 

**Key benefits**:
- Straightforward implementation
- Consistent chunk sizes
- Easily adaptable to different model requirements

**Types**:

- Token-based: Splits text based on the number of tokens, which is useful when working with language models.
- Character-based: Splits text based on the number of characters, which can be more consistent across different types of text.

**Character based example**

```python
from langchain.text_splitter import CharacterTextSplitter
    RecursiveCharacterTextSplitter,

c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
text = 'abcdefghijklmnopqrstuvwxyz'
c_splitter.split_text(text)
r_splitter.split_text(text)
```

- `RecursiveCharacterTextSplitter`
    - `seperators: list[str]`: splitter will try to split text using an element of `seperators` from first to last order until the chunk size becomes smaller than `chunk_size`.

**Token based example**

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
```

## Text-structured based

Common text is formed in hierarchical structure unit such as paragraphs, sentences, and words. With `RecursiveCharacterTextSplitter`, the text can be splitted into each unit to maintain the sementic meaning.

`RecursiveCharacterTextSplitter` achiev this by :
- attempting to keep larger units (i.e. paragraphs).
- If a unit exceeds the chunk size, it moves to the next level (i.e. sentences).
- This process continues down to the word level if necessary.

**RecursiveCharacterTextSplitter example**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "\. ", " ", ""]

)
docs = text_splitter.split_documents(pages)
```

## Document-structured based

Certain document such as HTML, Markdown or JSON can be splitted into semantically related text based on their format.

**Key benefits**

- Preserves the logical organization of the document
- Maintains context within each chunk
- Effective for downstream tasks like retrieval or summarization

**Examples of structure-based splitting:**

- Markdown: Split based on headers (e.g., #, ##, ###)
- HTML: Split using tags
- JSON: Split by object or array elements
- Code: Split by functions, classes, or logical blocks

## Semantic meaning based

Unlike the previous methods, semantic-based splitting actually considers the content of the text. While other approaches use document or text structure as proxies for semantic meaning, this method directly analyzes the text's semantics. There are several ways to implement this, but conceptually the approach is split text when there are significant changes in text meaning. As an example, we can use a sliding window approach to generate embeddings, and compare the embeddings to find significant differences:

- Start with the first few sentences and generate an embedding.
- Move to the next group of sentences and generate another embedding (e.g., using a sliding window approach).
- Compare the embeddings to find significant differences, which indicate potential "break points" between semantic sections.

This technique helps create chunks that are more semantically coherent, potentially improving the quality of downstream tasks like retrieval or summarization.