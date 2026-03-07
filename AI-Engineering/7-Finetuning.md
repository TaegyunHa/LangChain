# Finetuning

## 1. Overview

Finetuning is a form of **transfer learning** — it takes a pre-trained model and further trains it (updates its weights) so it performs well on a new, specific task. For example, a base model trained for general text completion can be finetuned to become a text-to-SQL converter.

Three main reasons to finetune:
- **Domain-specific capability** — improve performance on specialised tasks (e.g. medical question answering, coding)
- **Safety** — reduce harmful or biased outputs
- **Instruction-following** — ensure the model adheres to specific output styles, formats, and tool-calling conventions

Finetuning can also extend a model's context length through **long-context finetuning**.

> **Key Insight**: The finetuning process itself isn't hard — many frameworks handle training with sensible defaults. The complexity lies in deciding *whether* to finetune, preparing the data, and maintaining the model afterwards.

> Transfer learning is widely used in computer vision. Lower-level weights (detecting basic patterns like lines and curves) are kept, while higher-level weights (detecting complex patterns like shapes and objects) are retrained.

---

## 2. When to Finetune

### 2.1 Progressive Workflow

Finetuning should be a **last resort**, not a first step. The recommended progression:

| Stage | Approach | Effort |
|-------|----------|--------|
| 1 | Prompt engineering | Low |
| 2 | Few-shot examples (~50) | Low |
| 3 | RAG (retrieval-augmented generation) | Medium |
| 4 | Advanced RAG (hybrid search, reranking) | Medium |
| 5 | Finetuning | High |
| 6 | Task decomposition | High |

Each stage requires more investment. Only move to the next when the previous stage proves insufficient.

### 2.2 Benefits

- Improves the model's quality — both general and task-specific capability
- Bias mitigation — reduce unwanted biases in outputs
- **Distillation** — a smaller finetuned model can imitate a larger model's behaviour, reducing cost

> According to [Grammarly's engineering blog](https://www.grammarly.com/blog/engineering/), their finetuned T5 models outperformed GPT-3 variants despite being 60 times smaller — demonstrating that strategic finetuning can beat general-purpose models.

### 2.3 Drawbacks

- **Performance degradation** on tasks outside the target domain
- **Resource intensive**, requiring:
  - High-quality annotated data
  - Expertise in model training
  - Infrastructure for self-hosting
  - Budget and policy for monitoring, maintaining, and updating the model

> It seems like a pretty large chunk of what tech companies do would be finetuning.

---

## 3. Finetuning vs RAG

> **Key Insight**: Use RAG for **facts**, use finetuning for **form**.

| Aspect | RAG | Finetuning |
|--------|-----|------------|
| **Purpose** | Provide missing or updated information | Fix output behaviour, format, or style |
| **Use when** | Model lacks knowledge or has outdated data | Model produces incorrect format or irrelevant outputs |
| **What it changes** | The input context (adds retrieved documents) | The model weights |
| **Analogy** | Giving someone a reference book | Teaching someone a new skill |

In practice, the two approaches are complementary. A model can be finetuned for a specific task format while using RAG to supply up-to-date information.

---

## 4. Memory Bottlenecks

Memory is typically the primary constraint when finetuning large models. Three factors determine the memory footprint:

1. **Parameter count** — total number of values in the model
2. **Trainable parameter count** — the subset of parameters that get updated during training
3. **Numeric precision** — how many bits represent each value (e.g. float32 uses 32 bits, float16 uses 16 bits)

> **Key Takeaway**: Most practitioners don't have enough hardware, time, or data for full finetuning of large models. This constraint has driven the development of memory-efficient techniques like PEFT and quantisation.

> TensorRT compilation can perform quantisation. In media domains, this concept matters too (pixel format).

---

## 5. Quantisation

Quantisation reduces memory usage by lowering the number of bits used to represent each value in the model.

| Type | Description | When applied |
|------|-------------|--------------|
| **Post-training quantisation (PTQ)** | Reduces precision after training is complete | Most common; well-supported by major frameworks |
| **Training quantisation** | Reduces precision during training itself | Emerging; reduces both inference and training cost |

PTQ is the more established and practical approach for most developers.

---

## 6. Parameter Efficient Finetuning (PEFT)

PEFT methods achieve strong performance while updating only a tiny fraction of the model's parameters, dramatically reducing memory requirements.

### 6.1 Adapter-Based Methods

These add small, trainable modules to the existing model while keeping the original weights frozen.

**LoRA (Low-Rank Adaptation)** is the dominant method:
- Instead of updating a full weight matrix, LoRA decomposes the update into **two smaller, lower-rank matrices**
- This drastically reduces the number of trainable parameters
- Key properties:
  - **Parameter-efficient** — far fewer values to train
  - **Data-efficient** — works well with limited training data
  - **Modular** — LoRA adapters are small, self-contained files that can be easily swapped, combined, or served alongside the base model
- Variants include **DoRA** and **qDoRA**, which further refine the approach

> I think LoRA is really clever. I never thought that a full matrix could be expressed by two lower-rank matrices, which reduces the parameter count significantly.

### 6.2 Soft Prompt-Based Methods

Instead of modifying model weights, these methods introduce **special trainable tokens** prepended to the input.

- The tokens are **not human-readable** — they are continuous vectors in the model's embedding space
- They are **learned during back-propagation**, not manually crafted
- Provides a middle ground between full finetuning and basic prompting

> I was curious if we could add tokens for training — for example, a type of conversation or expected output. The soft prompt approach is better because the model learns the patterns by itself.

---

## 7. Model Merging

Model merging combines multiple finetuned models into a single model that performs better than any of them individually.

### 7.1 Approaches

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **Summing** (delta vectors) | Add the weight differences between finetuned and base models | Simple; most common |
| **Layer stacking** | Stack layers from different models vertically | Preserves architecture |
| **Concatenation** | Combine models side by side | Significant memory overhead; generally discouraged |

### 7.2 Use Cases

- **On-device deployment** — merge specialised capabilities into a single compact model
- **Model upscaling** — combine strengths of multiple models
- **Multitask learning** — an alternative to traditional approaches

### 7.3 Connection to Multitask Learning

Traditional multitask approaches have limitations:
- **Simultaneous training** requires a carefully balanced dataset across all tasks
- **Sequential training** risks **catastrophic forgetting** — learning a new task overwrites what was learnt for previous tasks

Model merging offers a third path: finetune separate models on different tasks **in parallel**, then merge them. This avoids catastrophic forgetting and allows each model to be optimised independently.

> Is it possible to visualise the activated nodes and merge models based on the task? CNNs have this approach.

> Could conventional programming's DLL concept (modular libraries) be applied to models if they can be modularised?

---

## 8. Practical Decision Frameworks

### 8.1 Progression Path

1. Start with the **cheapest and fastest** model
2. Validate with a **mid-tier** model
3. Test with the **best available** model
4. Map the full **cost-performance frontier**
5. Select based on actual requirements

### 8.2 Distillation Path

1. Start with a small dataset and the **strongest affordable model**
2. Use the finetuned model to **generate expanded training data**
3. Train a **cheaper, smaller model** on the augmented dataset

> **Key Takeaway**: Finetuning demands organisational commitment beyond technical implementation — it requires evaluating business priorities, resource availability, and long-term maintenance capacity.

---

## 9. Glossary of Key Terms

| Term | Definition |
|------|-----------|
| **PEFT** | Parameter Efficient Finetuning — methods that update only a small fraction of model parameters |
| **LoRA** | Low-Rank Adaptation — decomposes weight updates into two smaller matrices |
| **DoRA / qDoRA** | Variants of LoRA with refined weight decomposition |
| **Quantisation** | Reducing the number of bits used to represent model values |
| **PTQ** | Post-Training Quantisation — quantising a model after training |
| **Distillation** | Training a smaller model to imitate a larger model's outputs |
| **Catastrophic forgetting** | When learning new tasks overwrites previously learnt capabilities |
| **Delta vector** | The difference between a finetuned model's weights and the base model's weights |
| **Soft prompt** | Trainable, continuous tokens prepended to model input (not human-readable) |
| **Prompt caching** | Reusing previously computed representations for repetitive prompt segments |
| **Trainable parameter** | A parameter that gets updated during finetuning |
| **Loss** | The difference between the model's output and the expected output |
| **Gradient** | How much each trainable parameter contributes to the error |
| **Optimiser** | Algorithm that determines how much to adjust each parameter (e.g. Adam, SGD) |
