# LangChain Study Note

This repository will provide a summary of LangChain courses by DeepLearning.AI.

## Covered Courses

| Course                                    | Study Note                                                                   | Link                                                                                    |
| ----------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| LangChain for LLM Application Development | [Summary](./LangChain_for_LLM_Application_Development/7_Summary.md)          | [Course](https://learn.deeplearning.ai/courses/langchain)                               |
| LangChain Chat with Your Data             | [Summary](./LangChain_Chat_with_Your_Data/5_Summary.md)                      | [Course](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data)           |
| Long-Term Agentic Memory With LangGraph   | [Summary](./LangGraph_Long_Term_Agentic_Memory/2_Baseline_Email_Asistant.md) | [Course](https://learn.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph) |

## Setup

### Python Environment

1. Install [uv](https://docs.astral.sh/uv/) if you haven't already:
   
    **macOS/Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    **Windows (PowerShell):**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

1. Install dependencies from the lockfile:
    ```bash
    uv sync
    ```

1. Activate the virtual environment:

    **macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```

    **Windows (PowerShell):**
    ```powershell
    .venv\Scripts\activate
    ```

### Environment Variables

Create a `.env` file in the root directory of this repository with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: The `.env` file should be placed in the root directory and will be automatically loaded by the Jupyter notebooks.

# License

[MIT License](./LICENSE)
