# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a study group repository containing educational materials for three major LangChain/LangGraph courses:

1. **LangChain for LLM Application Development** - Core LangChain concepts including models, prompts, chains, memory, QnA, evaluation, and agents
2. **LangChain Chat with Your Data** - RAG (Retrieval Augmented Generation) focused course covering document loading, splitting, vectorstores, embeddings, retrieval, and chat systems
3. **LangGraph Long Term Agentic Memory** - Advanced agentic workflows with memory management using LangGraph

## Repository Structure

Each course has:
- **Markdown notes** (numbered files like `1_Document_Loading.md`, `2_memory.md`)  
- **Jupyter notebooks** in `jupyter/` subdirectories with executable code examples
- **Summary files** (`7_Summary.md`, `LangChain_Study_Summary.md`)

## Working with Notebooks

### Requirements
- Python environment uses requirements from `LangGraph_Long_Term_Agentic_Memory/jupyter/requirements.txt`
- Key dependencies: `langchain==0.3.18`, `langchain-openai==0.3.5`, `langchain-anthropic==0.3.7`, `langgraph==0.2.72`, `langmem==0.0.8`
- Requires API keys in `.env` file for OpenAI/Anthropic services

### Running Notebooks
- Navigate to appropriate `jupyter/` directory within each course folder
- Notebooks are designed to be run sequentially and build upon each other
- Some notebooks require external resources (PDFs, YouTube videos, web content) in `docs/` folders

## Key Learning Concepts by Course

### LangChain for LLM Application Development
- **Models**: LLMs, Chat Models, Text Embedding Models
- **Prompts**: ChatPromptTemplate, Output Parsers, ResponseSchema
- **Memory**: ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
- **Chains**: LLMChain, Sequential chains, Router chains
- **Agents**: Tools, ReAct framework, agent executors

### LangChain Chat with Your Data  
- **Document Loading**: PyPDFLoader, WebBaseLoader, YoutubeAudioLoader, NotionDirectoryLoader
- **Text Splitting**: RecursiveCharacterTextSplitter, token-based splitting
- **Vector Stores**: Chroma, FAISS integration with embeddings
- **Retrieval**: Similarity search, MMR, compression, contextual compression
- **QnA**: RetrievalQA, ConversationalRetrievalChain

### LangGraph Long Term Agentic Memory
- **State Management**: StateGraph, persistent state across interactions  
- **Agent Workflows**: Email triage, routing, response generation
- **Tools Integration**: Calendar, email, scheduling tools
- **Memory**: Long-term context preservation across sessions

## Development Notes

- This is a study/reference repository - no build or test commands needed
- Jupyter notebooks are self-contained with their own setup cells
- External API calls require proper authentication setup
- Some notebooks may take time to complete due to external service calls (YouTube, web scraping)