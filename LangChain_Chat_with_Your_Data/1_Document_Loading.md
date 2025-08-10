# Document Loading

# LangChain Objects

- `langchain.document_loaders.PyPDFLoader`
- `langchain.document_loaders.generic.GenericLoader`
- `langchain.document_loaders.generic.FileSystemBlobLoader`
- `langchain.document_loaders.parsers.OpenAIWhisperParser`
- `langchain.document_loaders.blob_loaders.youtube_audio.YoutubeAudioLoader`
- `langchain.document_loaders.WebBaseLoader`
- `langchain.document_loaders.NotionDirectoryLoader`

# General Process

Retrieval Augmented Generation

- Vector Store Loading
    1. Document Loading
        - URLs, DB -> Documents
    2. Splitting
        - Splits documents into chunks 
    3. Storage
        - Vector Store
- Retrieval
    1. Question Query
    2. Storage
        - Vector Store
    3. Retrieval
        - Find relevant splits
    4. Output
        - Prompt -> LLM -> Answer

# Loaders

Loaders can access different data sources and convert data
- Loader Input
    - Access
        - Web Site
        - Data Base
        - YouTube
        - arXiv
    - Data Format
        - PDF
        - HTML
        - JSON
        - Word
- Loader Output
    - `Document`

**PDF Example**
```python
from langchain.document_loaders import PyPDFLoader

pdf_path = "~/Documents/my_doc.mdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
# content access
pages[0].page_content
# metadata access
pages[0].metadata
```

**YouTube Example**
```python
langchain.document_loaders.generic import GenericLoader
langchain.document_loaders.parser import OpenAIWhisperParser
langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=Tc_jntovCM0"
save_dir="docs/youtube/"
loader = GenericLoader(
    #YoutubeAudioLoader([url],save_dir),  # fetch from youtube
    FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
    OpenAIWhisperParser()
)
docs = loader.load()
```

**URL Example**
```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()
```

**Notion Example**
```python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
```
