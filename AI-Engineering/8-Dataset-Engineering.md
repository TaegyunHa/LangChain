# Chapter 8: Dataset Engineering

## Executive Summary

- The goal of dataset engineering is to create the best training dataset within a given budget. Start by thinking through the behaviours you want the model to learn, then design a dataset that demonstrates those behaviours.
- Three criteria guide dataset construction: **quality**, **coverage**, and **quantity** — with quality being the most impactful.
- Synthetic data is a practical solution for scaling datasets, but it requires the same rigorous evaluation as real data and cannot fully replace human-generated data.

---

## Data Curation

Data curation is about understanding how the model learns and what resources are available to learn from.

The trend has shifted from **model-centric AI** to **data-centric AI**.
- **model-centric AI**: best model given fixed data
- **data-centric AI**: best dataset given a fixed model

### What data is needed?

Different training stages require different data types:

| Training Type | Data Format | Example |
|---|---|---|
| Self-supervised | Sequence of tokens | Raw text corpora |
| Instruction fine-tuning | `(instruction, response)` | Question-answer pairs |
| Preference fine-tuning | `(instruction, winning response, losing response)` | RLHF training data |

Special considerations for specific capabilities:
- **Chain of thought (CoT)**:
  - Include step-by-step reasoning in responses during fine-tuning, so the model learns to reason rather than just output answers.
- **Tool use**:
  - Show examples of how tools are selected and used. Note that conversation data alternates turns between user and AI, while tool use data often has the AI handling multiple messages per turn.

### Data Quality

Small, high-quality datasets outperform large, noisy ones. High quality means the data helps the model do its job efficiently and reliably.

Quality criteria:
- **Relevant**: directly related to the target task
- **Aligned with requirements**: matches the desired style (creative, factual, etc.)
- **Consistent**: annotations are uniform across examples and annotators
- **Correctly formatted**: follows the format the model expects
- **Sufficiently unique**: more unique examples lead to better training
- **Compliant**: meets legal and regulatory requirements

### Data Coverage

Training data should cover the range of problems the model is expected to solve.

- Greater diversity in data improves model performance
- Code often represents ~50% of LLM training data — even for non-coding models — because it improves reasoning ability
- Even 1% representation of a language can enable meaningful capabilities in that language

### Data Quantity

Pre-training can freeze the model's existing knowledge, making it harder to teach new behaviours. Evaluate whether to train from scratch or fine-tune a pre-trained model.

| Factor | Less Data Needed | More Data Needed |
|---|---|---|
| Fine-tuning technique | PEFT (few examples) | Full fine-tuning (many examples) |
| Task complexity | Simple tasks (e.g. classification) | Complex tasks (e.g. question answering) |
| Base model quality | Strong base model (~100 examples) | Weak base model |

There are diminishing returns — after a certain point, adding more data yields smaller improvements.

**Recommended progression**: use lower-quality data first, then refine:
1. Self-supervised → supervised
2. Less-relevant → relevant
3. Synthetic → real

> **Key Takeaway: Data Curation**
> - Quality matters most: design data around target behaviours
> - Ensure coverage across the full range of expected tasks
> - Start with available data and progressively improve quality

---

## Data Acquisition and Annotation

The goal is to produce a sufficiently large dataset with the right quality and diversity.

A practical pipeline example from the book:
1. Start with ~10,000 examples
2. Filter out low-quality instances → ~9,000
3. Remove poor responses → ~6,000
4. Manually write high-quality responses
5. Identify topic gaps using ~100 templates
6. Synthetically generate ~2,000 instructions
7. Manually annotate the synthetic data
8. **Result**: ~11,000 high-quality examples

> **Key Takeaway: Data Acquisition**
> - Data acquisition is iterative, not linear — expect to go back and forth between stages
> - Combine manual annotation with synthetic generation for scale and quality

---

## Data Augmentation and Synthesis

The goal is to generate data programmatically. Two common approaches:
- **Data augmentation**: create new data from existing data (e.g. paraphrasing, cropping)
- **Data synthesis**: create new data that mimics real data

### Why data synthesis?

- **Quantity**: produce data at scale
- **Coverage**: generate data with targeted characteristics to fill gaps
- **Quality**: AI-generated data can sometimes exceed human quality (e.g. tool use data, complex maths)
- **Privacy**: remove sensitive personal information
- **Distillation**: train a smaller model to imitate a larger model's behaviour

### Traditional Data Synthesis Techniques

**Procedural generation**: using algorithms to create data.

- **Rule-based**
  - Template structures following grammar and syntax rules
  - Simple transformations to mitigate bias
  - Perturbation: adding noise to existing data
- **Simulation**
  - Simulate experiments virtually at minimal cost
  - Avoid physical accidents and damage

### AI-Powered Data Synthesis

- AI can simulate the outcomes of arbitrary programs
- **Self-play**: a model learns by playing against itself
- **Back-translation**: verify translation quality by translating back and comparing
- **Reverse instruction**: take existing high-quality content, then use AI to generate prompts that would produce such content

Synthetic data is primarily used in **post-training** (fine-tuning), not pre-training. Pre-training is about building knowledge, and it is harder to synthesise genuinely new knowledge.

**Example**: The LLaMA 3.1 pipeline generated 2.7 million synthetic coding examples for supervised fine-tuning through problem generation, multi-language solutions, unit tests, error correction, and conversation generation.

### Limitations of AI-Generated Data

AI-generated data will never entirely replace human-generated data:
- **Quality gap**: subtle differences in naturalness and depth
- **Imitation limitations**: struggles with factual accuracy and generalisation; prone to hallucination
- **Model collapse**: repeated use of synthetic data can cause irreversible defects — forgetting rare events and amplifying biases
- **Lineage**: obscures the origin and provenance of the data

### Model Distillation

Model distillation is a method where a smaller model is fine-tuned on the outputs of a larger, more capable model. The smaller model learns to reproduce the larger model's behaviour at a fraction of the inference cost. This is a common and effective post-training strategy.

> **Key Takeaway: Data Synthesis**
> - Synthetic data is powerful for scaling, but requires the same quality checks as real data
> - Pre-training favours real data; post-training benefits most from synthetic data
> - Watch for model collapse when using synthetic data repeatedly

---

## Data Processing

Optimise efficiency during processing:
- Remove duplicates **before** cleaning
- Run trial validations before full processing
- Never change data in place. always make copies

### Inspect Data

Before any processing, manually inspect the data. As an OpenAI co-founder observed: manual inspection of data has probably the highest value-to-prestige ratio of any activity in machine learning. Spend at least 15 minutes directly examining examples before starting automated processing.

### Deduplicate Data

Duplicate data has negative effects:
- Skews data distribution
- Introduces bias
- Contaminates the test set

Deduplication methods:
- **Pairwise comparison**: compute similarity scores between examples
- **Hashing**: group potential duplicates into the same bucket
- **Dimensionality reduction**: reduce data dimensions and compare in lower-dimensional space

### Clean and Filter Data

Data needs to be cleaned for both performance and safety:
- Remove irrelevant formatting tokens (e.g. HTML tags)
- Filter content that is not compliant with policies
- Remove low-quality examples
- **Active learning**: select the examples most helpful for the model to learn from

### Format Data

Get data into the format the model expects:
- Fine-tuning can help manage cost (e.g. shorter inputs)
- Ensure the prompt format used at inference matches the format used during fine-tuning

> **Key Takeaway: Data Processing**
> - Always inspect data manually before automating
> - Order operations efficiently: deduplicate → clean → format
> - Preserve original data — work on copies

---

## Summary

Dataset engineering is an iterative process across three pillars: **curation**, **augmentation/synthesis**, and **processing**. The quality of a model depends on the quality of its training data. The best team with unlimited compute cannot fine-tune a good model without good data. Start by defining the behaviours you want the model to learn, design a dataset that demonstrates those behaviours, and progressively refine through cycles of inspection, synthesis, and cleaning.
