# Evaluation

## Key LangChain Objects

- `langchain.evaluation.qa.QAGenerateChain`
- `langchain.evaluation.qa.QAEvalChain`
- `langchain.chains.RetrievalQA`
- `langchain.indexes.VectorstoreIndexCreator`
- `langchain.debug`

## Overview

Evaluation is critical for ensuring LLM applications produce reliable and useful outcomes across diverse inputs. LangChain provides comprehensive evaluation tools for assessing model performance, particularly for question-answering systems.

**Evaluation Process**:
1. **Example Generation**: Create test datasets manually or automatically
2. **Manual Evaluation**: Debug and analyze individual responses
3. **LLM-Assisted Evaluation**: Scale evaluation using automated grading
4. **Performance Analysis**: Compare predictions against expected answers

## Example Generation

### Hard-coded Examples

Manual creation of question-answer pairs for testing specific scenarios:

```python
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty 850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
```

### LLM-Generated Examples

Automated example generation using QAGenerateChain for scalable test dataset creation:

```python
from langchain.evaluation.qa import QAGenerateChain
from langchain.chat_models import ChatOpenAI

# Create generation chain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))

# Generate examples from documents
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

# Combine with manual examples
examples += new_examples
```

**Benefits of LLM-Generated Examples**:
- Saves human effort in dataset creation
- Scales evaluation process efficiently
- Generates diverse question types
- Maintains consistency across examples

**Important**: Human oversight remains crucial for ensuring quality of LLM-generated datasets.

## Manual Evaluation and Debugging

### Debug Mode

Enable detailed logging to understand chain execution flow:

```python
import langchain
langchain.debug = True

# Run query with detailed output
qa.run(examples[0]["query"])

# Turn off debug mode
langchain.debug = False
```

**Debug Output Provides**:
- Step-by-step chain execution
- Intermediate results and reasoning
- Token usage and timing information
- Error detection and troubleshooting

### Manual Analysis

Direct examination of individual responses for quality assessment:
- Compare generated answers with expected results
- Analyze reasoning and context usage
- Identify common failure patterns
- Validate retrieval accuracy

## LLM-Assisted Evaluation

### QAEvalChain

Automated evaluation using LLM to grade question-answering performance:

```python
from langchain.evaluation.qa import QAEvalChain

# Generate predictions for evaluation
predictions = qa.apply(examples)

# Create evaluation chain
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
eval_chain = QAEvalChain.from_llm(llm)

# Evaluate predictions against examples
graded_outputs = eval_chain.evaluate(examples, predictions)
```

### Evaluation Analysis

Comprehensive assessment of model performance:

```python
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```

**Evaluation Metrics Include**:
- **Correctness**: Accuracy of factual information
- **Completeness**: Coverage of expected answer components
- **Relevance**: Appropriateness to the specific question
- **Consistency**: Reliability across similar queries

## QA Application Setup

### Complete Evaluation Pipeline

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

# Load and index documents
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# Create QA chain
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
```

## Evaluation Best Practices

### Test Dataset Design

**Comprehensive Coverage**:
- Include various question types (factual, comparative, complex)
- Test edge cases and boundary conditions
- Cover different document sections and topics
- Include both simple and multi-step reasoning

**Quality Assurance**:
- Validate ground truth answers for accuracy
- Ensure questions are clear and unambiguous
- Test with domain experts when possible
- Regular dataset updates and maintenance

### Evaluation Metrics

**Quantitative Measures**:
- **Accuracy Rate**: Percentage of correct answers
- **Response Time**: Average query processing time
- **Retrieval Precision**: Relevance of retrieved documents
- **Coverage**: Percentage of questions successfully answered

**Qualitative Assessment**:
- **Response Quality**: Coherence and clarity of answers
- **Context Usage**: Effective use of retrieved information
- **Error Types**: Classification of failure modes
- **User Satisfaction**: End-user feedback and ratings

### Continuous Evaluation

**Monitoring Framework**:
- Automated evaluation pipelines
- Regular performance benchmarking
- A/B testing for model improvements
- Production monitoring and alerting

**Iterative Improvement**:
- Analyze failure patterns for system enhancement
- Update training data based on evaluation results
- Refine retrieval strategies and parameters
- Optimize prompt engineering and chain configuration

## Use Cases

**Application Development**:
- Pre-deployment quality assurance
- Model comparison and selection
- Feature development validation
- Performance regression testing

**Production Monitoring**:
- Real-time response quality assessment
- User experience optimization
- System performance tracking
- Continuous improvement feedback loops

**Research and Development**:
- Experimental model evaluation
- Benchmark dataset creation
- Algorithm comparison studies
- Academic research validation