# Introduction to Building AI Applications with Foundation Models

## Overview

**AI Engineering** has emerged as a distinct discipline focused on building applications using pre-trained foundation models rather than training models from scratch.

This chapter provides:
- **Foundation model evolution** from self-supervised language models to multimodal systems, creating the "model-as-a-service" approach
- **Eight major use case categories** organised by risk level, from internal coding tools (low risk) to customer-facing AI companions (high risk)
- **Product-first development approach** using existing models before investing in custom solutions, fundamentally different from traditional ML engineering
- **Practical frameworks** including Microsoft's Crawl-Walk-Run for gradual automation and the 80/20 rule for managing development expectations

---

## 1. Foundation Models - The Building Blocks

### 1.1 The Emergence of Foundation Models

> **Key Insight**: Scale + self-supervision = AI accessible to everyone

Foundation models represent a major change from task-specific models to general-purpose systems. The journey began with a simple observation: language follows statistical patterns that can be learned without manual labelling.

#### Historical Context & Milestones

| Year | Breakthrough | Impact | Parameters |
|------|-------------|---------|------------|
| **2017** | Transformer Architecture | Enabled parallel training at scale | ~100M |
| **2018** | BERT | Bidirectional understanding | 340M |
| **2020** | GPT-3 | Emergent capabilities at scale | 175B |
| **2021** | CLIP | First major multimodal model | 400M |
| **2022** | ChatGPT | Mainstream AI adoption | ~20B |
| **2023** | GPT-4 | Native multimodal integration | ~1.76T |
| **2024** | Claude 3, Gemini | Long context, video understanding | Undisclosed |

What made this possible: **self-supervision** removed the need for manual labelling, allowing models to learn from internet-scale data.

### 1.2 Core Technical Concepts

#### Tokenisation: The Fundamental Unit

Tokens balance three critical requirements:
1. **Semantic meaning** (unlike individual characters)
2. **Vocabulary efficiency** (fewer unique tokens than words)
3. **Flexibility** (can handle unknown words)

**Quick Reference:**
- Average token ≈ 3/4 of a word
- Vocabulary size: typically 30,000-100,000 tokens
- Special tokens: `<BOS>` (beginning), `<EOS>` (end)

#### Self-Supervision: The Training Revolution

| Traditional Supervised Learning | Self-Supervised Learning |
|--------------------------------|--------------------------|
| Requires labelled data | Derives labels from input |
| Expensive annotation | Zero annotation cost |
| Limited scale | Internet-scale training |
| Task-specific | General-purpose capabilities |

**How it works**: Each word in a sequence becomes both context and label:
```
Input: "I love food"
Training pairs:
  (<BOS>) → I
  (<BOS>, I) → love
  (<BOS>, I, love) → food
```

### 1.3 Model Types and Capabilities

#### Language Model Categories

| Type | Masked (BERT-style) | Autoregressive (GPT-style) |
|------|-------------------|---------------------------|
| **Training** | Predict missing tokens | Predict next token |
| **Context** | Bidirectional | Left-to-right only |
| **Best for** | Understanding, classification | Generation, completion |
| **Examples** | BERT, RoBERTa | GPT, Claude, Llama |

#### Evolution to Multimodality

**Foundation Model** = Language Model + Vision + Audio + ...

The breakthrough: CLIP showed that different types of data (text, images) can share the same representation space, allowing:
- Image generation from text (DALL-E, Midjourney)
- Video understanding (Gemini)
- Document processing with layout awareness (Claude)

---

## 2. Adaptation Techniques - Making Models Work for You

### 2.1 The Adaptation Hierarchy

Choose your technique based on data availability and performance requirements:

#### Decision Framework

| Data Available | Technique | Time to Deploy | Cost      |
|----------------|-----------|----------------|-----------|
| 0-10 examples  | Prompting | Hours          | Low       |
| 100s docs      | RAG       | Days           | Medium    |
| 1000s examples | Fine-tune | Weeks          | High      |
| Millions       | Training  | Months         | Very High |

### 2.2 Detailed Technique Comparison

| Aspect | **Prompt Engineering** | **RAG** | **Fine-tuning** |
|--------|----------------------|---------|-----------------|
| **Setup complexity** | Minimal | Moderate | High |
| **Iteration speed** | Minutes | Hours | Days |
| **Maintenance** | Update prompts | Update documents | Retrain model |
| **Performance ceiling** | ~70-80% | ~85-90% | ~95%+ |
| **Context limit** | Model's window | Retrieved chunks | Embedded in weights |
| **Use cases** | Prototypes, simple tasks | Knowledge-intensive | Style, format, domain |

### 2.3 Current Model Landscape (2024)

| Model | Context | Strengths | Limitations | Best For |
|-------|---------|-----------|-------------|----------|
| **GPT-4** | 128K | General reasoning, code | Cost, latency | Complex analysis |
| **Claude 3** | 200K | Document processing | Availability | Long documents |
| **Gemini Ultra** | 1M | Video, multimodal | Beta features | Media processing |
| **Llama 3** | 8K | Open source | Smaller context | Self-hosting |
| **Mistral Large** | 32K | EU compliance | Ecosystem | European projects |

> **Key Takeaway**: Start with prompt engineering. Only invest in more complex techniques when you hit clear limitations.

---

## 3. Use Cases - Where AI Creates Value

### 3.1 Risk-Based Implementation Strategy

Start with low-risk internal applications, then gradually expand:

#### Tier 1: Low Risk (Start Here)
**Characteristics**: Internal-facing, human oversight, reversible decisions

| Use Case | Examples | Success Metrics | Typical ROI |
|----------|----------|----------------|-------------|
| **Coding** | Copilot, documentation | 46% code completion | 57% faster |
| **Information** | Research, summaries | 10x faster review | 70% time saved |
| **Data Organisation** | Categorisation, search | 90% less manual work | 3.5x in 18 months |
| **Writing (Internal)** | Reports, emails | 40-60% time reduction | Immediate |

#### Tier 2: Medium Risk (Careful Testing)
**Characteristics**: Indirect customer impact, quality variations acceptable

| Use Case | Implementation Considerations |
|----------|------------------------------|
| **Education** | Start with employees before customers |
| **Workflow Automation** | Begin with non-critical processes |
| **Image/Video** | Internal content before public-facing |
| **Internal Chatbots** | IT support before customer service |

#### Tier 3: High Risk (Extensive Validation)
**Characteristics**: Customer-facing, regulatory implications, irreversible

| Use Case | Critical Requirements |
|----------|----------------------|
| **Customer Service** | Fallback to humans, quality monitoring |
| **Content Generation** | Brand safety, fact-checking |
| **AI Companions** | Ethical guidelines, safety measures |
| **Decision Support** | Audit trails, explainability |

### 3.2 Detailed Use Case Analysis

#### Coding: The Killer App
- **Market leader**: GitHub Copilot (1M+ developers)
- **Productivity gain**: 46% of code auto-completed
- **Key success factor**: Integration into existing workflows (IDEs)
- **Evolution**: From autocomplete → full application generation

#### Conversational AI: The Transformation Driver
- **Klarna**: 700 human agents replaced, 2/3 of chats automated
- **Morgan Stanley**: 16,000 advisors augmented
- **Character.AI**: 20M+ users for entertainment
- **Success pattern**: Start internal → validate → expand to customers

#### Creative Production: Making Creativity Accessible
- **Midjourney**: $200M ARR, 15M users
- **Adobe Firefly**: 3B images generated
- **Enterprise adoption**: Coca-Cola, Nutella, Netflix campaigns
- **Key insight**: AI for ideation and drafts, humans for refinement

---

## 4. Building AI Applications - From Concept to Production

### 4.1 The Crawl-Walk-Run Framework

Microsoft's proven approach for gradual AI automation:

| Stage | **Crawl** | **Walk** | **Run** |
|-------|----------|---------|---------|
| **AI autonomy** | None - suggests only | Internal interactions | External interactions |
| **Human role** | Makes all decisions | Validates decisions | Monitors exceptions |
| **Risk level** | Minimal | Moderate | Managed |
| **Example** | Code suggestions | Employee chatbot | Customer service |
| **Timeline** | Weeks | Months | Quarters |

### 4.2 The 80/20 Reality

```
Progress vs Effort Distribution:
┌─────────────────────────────────┐
│ 0% → 60%:  20% effort (2 weeks) │ ← Basic functionality
│ 60% → 80%: 20% effort (2 weeks) │ ← Good enough for internal
│ 80% → 90%: 30% effort (3 weeks) │ ← Production-ready
│ 90% → 100%: 30% effort (3 weeks)│ ← Perfection (rarely needed)
└─────────────────────────────────┘
```

**Implications**:
- Demo ≠ Product
- Budget 5x time for the "last mile"
- Consider if 80% quality is sufficient

### 4.3 Pre-Flight Checklist

#### Technical Requirements
- [ ] Response time requirements defined
- [ ] Expected request volume calculated
- [ ] Privacy and compliance reviewed
- [ ] Integration points identified

#### Success Metrics
- [ ] Primary KPIs identified
- [ ] Baseline performance measured
- [ ] Acceptable error rates defined
- [ ] Fallback mechanisms designed

#### Resource Assessment
- [ ] Budget approved for dev + operations
- [ ] Team skills gap analysis complete
- [ ] Timeline with buffers established
- [ ] Risk mitigation plan documented

### 4.4 Building Competitive Advantage

When it's easy for others to copy your work, protection comes from:

| **Moat Type** | **Examples** | **Sustainability** |
|---------------|--------------|-------------------|
| **Data** | User interactions, feedback loops | High - compounds |
| **Distribution** | Existing users, channels | Medium - replicable |
| **Technology** | Proprietary models, techniques | Low - quickly becomes common |
| **Integration** | Workflow embedding | High - switching cost |
| **Brand** | Trust, reputation | Medium - takes time |

**The Data Flywheel**: Better data → Better models → More users → More data

---

## 5. The AI Stack - Architecture and Roles

### 5.1 Three-Layer Architecture

| Layer | Responsibilities | Primary Role |
|-------|-----------------|--------------|
| **Application Layer** | Evaluation, Prompts, UX | AI Engineers |
| **Model Layer** | Training, Fine-tuning, Optimisation | ML Engineers |
| **Infrastructure Layer** | Serving, Monitoring, Scaling | Platform Engineers |

### 5.2 AI Engineering vs ML Engineering

The fundamental change in development approach:

| **Aspect** | **Traditional ML** | **AI Engineering** |
|------------|-------------------|-------------------|
| **Starting point** | Data | Product |
| **Model source** | Train from scratch | Pre-trained |
| **Iteration** | Weeks-months | Hours-days |
| **Primary skill** | Statistics | System design |
| **Output** | Predictions | Generation |
| **Evaluation** | Metrics | Human preference |
| **Cost structure** | Upfront training | Pay-per-use |
| **Team size** | 5-10 specialists | 1-3 generalists |

### 5.3 Essential AI Engineering Skills

#### Core Competencies
1. **Prompt Engineering** - Crafting effective instructions
2. **Evaluation Design** - Measuring open-ended outputs
3. **Context Management** - Handling memory and state
4. **Interface Design** - Creating intuitive AI interactions
5. **System Architecture** - Orchestrating AI components

#### The Full-Stack AI Engineer
- Frontend: Chat interfaces, streaming responses
- Backend: API integration, context management
- Data: Vector stores, embeddings
- Operations: Monitoring, cost optimisation
- Product: User research, iteration

---

## Decision Frameworks

### When to Use AI - Decision Tree

```
Should you use AI for this task?
│
├─ Can a rule-based system solve it?
│   └─ Yes → Don't use AI
│
├─ Is the task creative or open-ended?
│   └─ Yes → Good fit for AI
│
├─ Do you need 100% accuracy?
│   └─ Yes → AI alone insufficient
│
├─ Is human-level performance acceptable?
│   └─ Yes → Consider AI
│
└─ Can errors be easily corrected?
    └─ Yes → AI with human oversight
```

### Build vs Buy vs Adapt

| **Scenario** | **Recommendation** | **Rationale** |
|-------------|-------------------|---------------|
| Generic use case | Buy (API) | Common, no unique advantage |
| Domain-specific | Adapt (RAG/Fine-tune) | Balance of customisation and effort |
| Core competency | Build | Strategic advantage |
| Experimental | Start with API | Validate before investing |

---

## Glossary of Key Terms

**AI Engineering** → Building applications using pre-trained foundation models

**Autoregressive** → Model that generates tokens sequentially based on previous tokens

**CLIP** → Contrastive Language-Image Pre-training; pioneering vision-language model

**Context Window** → Maximum tokens processable in single request (8K-1M tokens)

**Crawl-Walk-Run** → Microsoft's framework for gradual AI automation

**Embedding** → Vector representation capturing semantic meaning

**Fine-tuning** → Continuing training on task-specific data

**Foundation Model** → Large-scale model adaptable to various tasks

**HITL** → Human-in-the-Loop; system with human oversight

**Last Mile Problem** → Disproportionate effort for final 20% quality

**Parameter** → Learnable model variable (millions to trillions)

**Prompt Engineering** → Crafting inputs for desired model behaviour

**Quantisation** → Reducing weight precision for efficiency

**RAG** → Retrieval Augmented Generation; combining search with generation

**Self-supervision** → Training where labels derive from input

**Token** → Basic text unit (~3/4 word)

**TPOT** → Time Per Output Token (generation speed)

**TTFT** → Time To First Token (initial response)

---

## Key Takeaways

> **Foundation Models**
> - Self-supervision enabled unprecedented scale
> - Evolution from text-only to multimodal systems
> - Model-as-a-service made AI accessible to everyone

> **Use Cases**
> - Start with low-risk internal applications
> - Eight major categories from coding to automation
> - Success requires matching use case to risk tolerance

> **Development Approach**
> - Product-first, not model-first
> - The 80/20 rule: last mile takes most effort
> - Crawl-Walk-Run for production deployment

> **AI Engineering**
> - Distinct from ML engineering
> - Focus on adaptation over training
> - Closer to full-stack than data science

> **Competitive Advantage**
> - Low barriers mean defensibility crucial
> - Data flywheel creates sustainable moat
> - Integration and distribution beat technology

---

## Next Steps

1. **Identify your first use case** using the risk framework
2. **Start with prompt engineering** before complex techniques
3. **Build a prototype** to validate assumptions
4. **Measure against the pre-flight checklist**
5. **Iterate using the Crawl-Walk-Run approach**

Remember: AI engineering is about building products, not models. Focus on user value, and the technology will follow.