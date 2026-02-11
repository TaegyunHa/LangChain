# Introduction to Building AI Applications with Foundation Models

## Executive Summary

This chapter introduces AI Engineering as a distinct discipline that emerged from the democratisation of foundation models. It covers:
- **Foundation Models**: Evolution from language models to multimodal systems through self-supervision
- **Use Cases**: Eight major application categories from coding to workflow automation
- **Planning & Development**: Frameworks for evaluating, building, and maintaining AI applications
- **AI Stack**: Three-layer architecture and the shift from ML to AI engineering paradigms

## Key Concepts

AI in one word post-2020: scale. The scaled-up AI models produced powerful and capable systems, but not affordable for independent developers. As a result, few organisations can afford them, which led to the emergence of **model as a service**. 

AI engineering: the process of building applications on top of model-as-a-service offerings from various organisations.

This chapter can be summarised in three sections:
1. Overview of Foundation models
   1. Covers why we want to use AI Engineering
2. Foundation model use cases
   1. Covers what can be done with AI Engineering
3. Planning AI Applications
   1. Covers how to do AI Engineering

## Overview of Foundation Models 

### History

Foundation models emerged from large language models through a series of breakthroughs:

#### Key Milestones in Foundation Model Development
- **2017**: Transformer architecture (Vaswani et al.) - enabled efficient parallel training
- **2018**: BERT - introduced bidirectional pre-training at scale
- **2020**: GPT-3 (175B parameters) - demonstrated emergent capabilities at scale
- **2021**: CLIP - bridged vision and language, first major multimodal model
- **2022**: ChatGPT - made LLMs accessible to mainstream users
- **2023**: GPT-4 - multimodal capabilities integrated natively
- **2024**: Claude 3, Gemini - continued advancement in reasoning and multimodality

LLMs have only been able to scale because of self-supervision, eliminating the need for manual labelling.

A language model encodes statistical information about language - how likely a word is to appear in a given context.
- Example: "My favourite colour is ___": blue (likely) over car (unlikely)

The statistical nature of language was discovered long ago:
- Sherlock Holmes stories demonstrated pattern recognition in language
- Claude Shannon's "Prediction and Entropy of Printed English" (1951) - mathematical foundation

### Token

- Token:
  - The basic unit of language model.
  - A token can be a character, a word, or part of a word (i.e. -tion)
- Tokenisation:
  - The process of breaking the original text into tokens
  - Average token is approximately 3/4 the length of a word
- Model's vocabulary:
  - The set of all tokens a model can work with.

Why LLM uses token as their unit over words or characters?
1. More meaningful to model
   1. alphabet doesn't have any meaning
2. Less unique token than words
   1. i.e. go, goes, going -> go, es, ing
3. Allow to process unknown words
   1. GPT-ing 

### Language models

There are two main types of language models
1. Masked language model
   1. Trained to predict missing token using the context from both before and after
2. Autoregressive language model
   1. Trained to predict the next token using only the preceding tokens
   2. Language model
   3. Generative: model can generate open-ended output

Language model is a completion machine, which can cover:
- translation, summarisation, coding and solving math problems.
- completion is not same as conversations
  - i.e. giving question as input
    - conversation: output is answer
    - non-conversation: additional question can be added in output

### Self-supervision

Language model could be scaled by self supervision.
- Supervision: trained using labelled data.
  - Expensive and time consuming
- self-supervision: trained without labelled data.
  - Model can infer label from input data.
  - Input sequence provides both label and the context the model can use to predict.

Input (context) | Output (next token)
--------|-
`<BOS>` | I
`<BOS>`, I | love
`<BOS>`, I, love | food
`<BOS>`, I, love, food | .
`<BOS>`, I, love, food, . | `<EOS>`

- `<BOS>`: Beginning of sequence
- `<EOS>`: End of sequence
  - Important because it helps language models know when to end the response

Parameter:
- A variable within an ML model that is updated through the training process
- Model size is measured by number of parameters

### From LLM to Foundation Models

Traditional model
- Natural language processsing (NLP) deals only with text
- Computer vision deals only with vision
- Audio model can handle speech recognition and synthesis

Multimodal model:
- Foundation model
- A model that can work with more than one data modality
- A generative multimodal model is called large multimodal model (LMM)
- Multimodal model generates the next token conditioned on both text and image token

I was working on computer vision, but it can be replaced by multimodal model.

CLIP:
- language model OpenAI trained using (image, text) pairs that co-occurred on the internet.
- First model to generalise to multiple image classification tasks.
- CLIP is an embedding model

Embedding: Vector that aim to capture the meaning of the original data.

### Model Adaptation Techniques Comparison

| Technique | Prompt Engineering | RAG | Fine-tuning |
|-----------|-------------------|-----|-------------|
| **What it is** | Crafting instructions and examples | Augmenting prompts with retrieved data | Further training the model |
| **Data needed** | 0-10 examples | 100s-1000s documents | 1000s-100,000s examples |
| **Cost** | Low ($0.01-0.1/request) | Medium ($0.02-0.2/request) | High (training + inference) |
| **Latency** | Baseline | +20-200ms | Baseline or better |
| **When to use** | Simple tasks, prototyping | Domain knowledge needed | Specific style/format needed |
| **Advantages** | Fast iteration, no training | Dynamic knowledge updates | Best quality, lower inference cost |
| **Disadvantages** | Limited by context window | Added complexity, retrieval errors | Requires expertise, can overfit |

### Popular Foundation Models Comparison (2024)

| Model | Parameters | Context Window | Strengths | Best For |
|-------|------------|----------------|-----------|----------|
| **GPT-4** | ~1.76T | 128K | General reasoning, coding | Complex tasks |
| **Claude 3** | Undisclosed | 200K | Long context, analysis | Document processing |
| **Gemini Ultra** | Undisclosed | 1M | Multimodal native | Video understanding |
| **Llama 3** | 8B-70B | 8K | Open source, customisable | Self-hosting |
| **Mistral Large** | 123B | 32K | European, multilingual | EU compliance needs |

- Considerations:
  - How much data is needed depends on what technique you use
  - Building own model vs Use existing model <-> Buy or build question in traditional business

### From Foundation Models to AI Engineering

> **Key Takeaways: Foundation Models**
> - Self-supervision enabled unprecedented scale in model training
> - Tokens are the fundamental unit, balancing meaning and efficiency
> - Evolution from text-only LLMs to multimodal foundation models
> - Three adaptation techniques: prompt engineering, RAG, and finetuning

AI engineering: Process of building applications on top of foundation models
- ML Engineering: Developing ML models
- AI Engineering: Leverages existing models.
- Three factors
  1. General purpose AI capabilities: 
     1. Model can do existing task better and more
     2. Help train even more powerful model in the future (self improving?)
  2. Increased AI investment
     1. AI applications become cheaper to build and faster to productise, which is attractive to investors.
  3. Low entrance barrier to build AI applications
     1. Easier to leverage AI to build applications
     2. Build applications with minimal coding


## Foundation Model Use Cases

AI usecases have been categorised in several way
- Amazon Web Services (AWS)
  - Customer experience
  - Employee productivity
  - Process optimisation
- Deloitte
  - cost reduction
  - process efficiency
  - growth
  - accelerating innovation
- O'Reilly
  - Programming
  - data analysis
  - Customer support
  - marketing copy
  - other copy
  - research
  - web designer
  - art

Eloundou et al. defined group of task as exposed if AI and AI-powered software can reduce the time needed to complete the task
- Human alpha: exposure to AI models directly
- Human beta, gamma: exposure to AI powered software

Use cases can be analysed through multiple lenses: consumer vs enterprise, and by risk level.

### Risk-Based Use Case Categorisation

#### Low Risk (Start Here)
Internal-facing applications with human oversight:
- **Coding assistance**: Code generation, refactoring, documentation
- **Information aggregation**: Summarisation, research, knowledge management
- **Data organisation**: Document processing, search, categorisation
- **Writing support**: Drafts, reports, internal documentation

#### Medium Risk (Careful Implementation)
Applications with indirect customer impact:
- **Education & Training**: Employee onboarding, upskilling programmes
- **Workflow automation**: Data entry, lead generation, process optimisation
- **Image/Video production**: Marketing materials, presentations
- **Internal chatbots**: IT support, HR queries, policy information

#### High Risk (Extensive Testing Required)
Direct customer-facing or critical decision-making:
- **Customer service bots**: External chatbots, complaint handling
- **Content generation**: Public-facing copy, SEO content
- **AI companions**: Mental health support, personal assistants
- **Decision support**: Loan approvals, medical diagnosis assistance

### Use Case Comparison Table

| Category | Consumer Examples | Enterprise Examples | Risk Level |
|----------|------------------|-------------------|------------|
| Coding | Personal projects | Production code | Low-Medium |
| Image/Video | Photo editing, art | Marketing, ads | Medium |
| Writing | Emails, blogs | Reports, documentation | Low-Medium |
| Education | Tutoring, language learning | Employee training | Medium |
| Conversation | AI companions | Customer support | High |
| Information | Personal research | Market intelligence | Low |
| Data Organisation | Personal knowledge base | Document management | Low |
| Workflow | Travel planning | Process automation | Medium |

Important to note that an application can belong to more than one category, and risk levels depend on implementation context.

### Coding

Coding is the most popular use case at the moment, with measurable productivity gains:

**Key Applications:**
- Documentation generation and code commenting
- Code generation from natural language descriptions
- Code refactoring and modernisation
- Bug detection and fixing
- Test case generation

**Real-World Impact:**
- **GitHub Copilot**: Used by over 1 million developers, completing 46% of code
- **Replit Ghostwriter**: Enables non-programmers to build functional applications
- **Amazon CodeWhisperer**: 57% faster task completion in productivity studies
- **Cursor**: IDE built around AI-first development paradigm

AI is disrupting the outsourcing industry while amplifying experienced developers' capabilities.

### Image and Video Production

AI has revolutionised creative workflows with both consumer and enterprise applications:

**Leading Platforms:**
- **Midjourney**: 15+ million users, $200M ARR (2023)
- **Adobe Firefly**: Integrated into Creative Cloud, 3 billion images generated
- **Runway**: Valued at $1.5B, powers Hollywood productions
- **Stable Diffusion**: Open-source, enabling countless applications
- **DALL-E 3**: Integrated with ChatGPT for seamless creation

**Enterprise Case Studies:**
- **Coca-Cola**: "Create Real Magic" campaign using AI-generated ads
- **Nutella**: 7 million unique jar designs using AI
- **Netflix**: AI-assisted thumbnail generation increasing click-through rates
- **WPP**: $30M AI creative platform investment

Enterprises use AI to generate promotional content, brainstorm concepts, and create first drafts for refinement.

### Writing

AI transforms writing from basic autocorrect to full content generation. ChatGPT notably closes the quality gap between workers, particularly benefiting those with less writing experience.

**Applications:** Email composition, blog posts, SEO content, reports, documentation, marketing copy
**Impact:** 40-60% time reduction in content creation, improved consistency across teams

### Education

AI enables personalised, scalable learning experiences through adaptive tutoring and content generation.

**Applications:** Personalised curricula, language learning (roleplay, quizzes), skill tutoring, employee training
**Impact:** 2x faster learning rates, 24/7 availability, adaptation to individual learning styles

### Conversational Bots

AI-powered conversational interfaces are transforming customer interactions:

**Applications:**
- Customer support automation
- Insurance claim processing
- Tax filing assistance
- Corporate policy guidance
- Smart NPCs in gaming

**Success Stories:**
- **Klarna**: AI assistant handling 2/3 of customer service chats, equivalent to 700 agents
- **Morgan Stanley**: Wealth management assistant serving 16,000 financial advisors
- **Duolingo Max**: AI tutor providing personalised language practice
- **Character.AI**: 20+ million users engaging with AI personalities
- **Epic Games**: Unreal Engine's MetaHuman conversational NPCs

### Information Aggregation & Data Organisation

AI excels at processing vast amounts of unstructured data and making it accessible.

**Information Aggregation:** Document processing (contracts, papers), talk-to-your-docs interfaces, research synthesis
**Data Organisation:** Semantic search, automatic categorisation, metadata extraction, knowledge graphs
**Combined Impact:** 10x faster document review, 90% reduction in manual data entry

### Workflow Automation

The ultimate goal: AI agents that can plan and execute complex multi-step tasks autonomously.

**Applications:** Lead management, invoice processing, customer request routing, data pipeline automation
**Key Technology:** Agents with tool-use capabilities (e.g., LangChain, AutoGPT, Custom agent frameworks)
**Impact:** 70% reduction in repetitive tasks, enabling humans to focus on high-value work

> **Key Takeaways: Use Cases**
> - Eight major categories: Coding, Image/Video, Writing, Education, Conversation, Information Aggregation, Data Organisation, Workflow Automation
> - Enterprises prefer lower-risk internal applications initially
> - Most successful applications belong to multiple categories
> - Coding is currently the most mature and popular use case

## AI Application

### Planning AI Applications

Building applications is the best way to learn, but production requires careful planning. Foundation models make demos easy but profitable products hard.

#### Use Case Evaluation - Three Risk Levels
1. **Existential**: Competitors with AI could make you obsolete
2. **Opportunity**: Missing profit-boosting capabilities
3. **Exploratory**: Uncertain fit but avoiding being left behind

### The Role of AI and Humans in Applications

**AI's Role Dimensions:**
- **Critical vs Complementary**: Does the app function without AI?
- **Reactive vs Proactive**: User-triggered (chatbot) vs system-initiated (alerts)
- **Dynamic vs Static**: Continuous learning (Face ID) vs periodic updates (photo tagging)

**Human-in-the-loop (HITL)**: Essential for high-stakes decisions, quality control, and continuous improvement

Microsoft's framework for gradually increasing AI automation in production: Crawl-Walk-Run:
1. Crawl: human involvement is mandatory
2. Walk: AI can directly interact with internal employee
3. Run: Increased automation, direct AI interactions with external users

### AI Product Defensibility
Low entry barrier is blessing and a curse.
- Easy for you to build, it's easy for your competitor to build.

Three types of competitive advantages:
- **Technology**: Proprietary models, unique architectures, or novel techniques
- **Data**:
  - First-mover advantage in collecting user interaction data
  - The data flywheel: better data → better models → more users → more data
  - Domain-specific datasets that are hard to replicate
- **Distribution**:
  - Existing user base and channels
  - Integration with established workflows
  - Brand trust and reputation

**Market Reality:**
- The Intelligent Document Processing market alone is projected to reach $12.81 billion by 2030
- Over 80% of enterprises are piloting or deploying AI applications (2024)
- Average AI project ROI: 3.5x within 18 months for successful implementations

### Setting Expectations

#### AI Project Pre-Flight Checklist

**□ Define Success Criteria**
- [ ] Identify primary success metrics (accuracy, user satisfaction, cost reduction)
- [ ] Set measurable targets (e.g., 90% accuracy, 50% cost reduction)
- [ ] Establish baseline performance for comparison
- [ ] Define failure modes and acceptable error rates

**□ Evaluate Technical Requirements**
- [ ] Required response time (real-time vs batch)
- [ ] Expected request volume
- [ ] Data privacy and compliance needs
- [ ] Integration requirements with existing systems

**□ Assess Resources**
- [ ] Budget for development and operations
- [ ] Team expertise (AI engineering, domain knowledge)
- [ ] Timeline and milestones
- [ ] Risk tolerance and fallback plans

**□ Choose Metrics Framework**

For a chatbot example:
- **Automation rate**: % of queries handled without human intervention
- **Throughput**: Messages processed per hour
- **Response time**: Average time to first response
- **Cost efficiency**: Cost per resolved query vs human baseline

Usefulness thresholds include following metric groups:
- **Quality metrics**: Accuracy, relevance, helpfulness ratings
- **Latency metrics**: TTFT (time to first token), TPOT (time per output token), total latency
- **Cost metrics**: Cost per inference request, TCO including infrastructure
- **Fairness metrics**: Bias detection, demographic parity

### Milestone Planning
Evaluate existing models to understand their capabilitities.
Goals will change after evaluation.
Planning an AI product needs to account for its last miles challenge.

#### The 80/20 Development Pattern
The journey from 0 to 60 is easy, whereas progressing from 60 to 100 becomes exceedingly challenging. This is known as the **80/20 rule** or **last mile problem** in AI development:
- **First 80%**: Relatively quick to achieve with existing models and basic prompting
- **Final 20%**: Often requires equal or greater effort, involving:
  - Extensive prompt engineering and iteration
  - Data collection and curation
  - Fine-tuning or model adaptation
  - Edge case handling and error recovery
- **Implications**: Plan resources accordingly - budget significant time for refinement

### Maintenance
Need to think about how the product might change over time, how it should be maintained.
Model inference, the process of computing an output given an input, is getting faster and cheaper.
Run a cost-benefit analysis. It's getting easier to swap one model API for another.

> **Key Takeaways: AI Application Planning**
> - Consider AI's role: critical vs complementary, reactive vs proactive, dynamic vs static
> - Use Microsoft's Crawl-Walk-Run framework for gradual automation
> - Low barriers to entry mean defensibility requires data, distribution, or technology advantages
> - The "last mile" from 60% to 100% quality is often the most challenging

## AI Stack
Important to recognise that AI engineering evolved out of ML engineering.
Position AI engineers and ML engineers, their roles have significant overlap.

### Three Layers of the AI Stack

The AI engineering ecosystem consists of three distinct layers:

1. **Application Development Layer**
   - Accessible to developers without ML expertise
   - Focus on evaluation, prompt engineering, and user interfaces
   - Tools: LangChain, Vercel AI SDK, Streamlit

2. **Model Development Layer**
   - Training, fine-tuning, and optimising models
   - Requires ML expertise and computational resources
   - Tools: PyTorch, TensorFlow, Hugging Face Transformers

3. **Infrastructure Layer**
   - Model serving, data management, compute orchestration
   - Monitoring, scaling, and operational concerns
   - Tools: Ray, Kubernetes, MLflow, Weights & Biases

### AI Engineering vs ML Engineering

Important to understand how the field has evolved:

| Aspect | Traditional ML Engineering | AI Engineering |
|--------|---------------------------|----------------|
| **Focus** | Model development | Application development |
| **Models** | Train from scratch | Use pre-trained foundation models |
| **Data Requirements** | Large labelled datasets | Few examples or unlabelled data |
| **Primary Skills** | Statistics, algorithms | Prompt engineering, system design |
| **Iteration Speed** | Weeks to months | Hours to days |
| **Output Type** | Structured predictions | Open-ended generation |
| **Evaluation** | Accuracy, F1, AUC | Human preference, task completion |
| **Infrastructure** | Training clusters | Inference endpoints |
| **Cost Structure** | High upfront (training) | Pay-per-use (inference) |

Building applications using foundation models today differs from traditional ML engineering:
1. Use models someone else has trained for you
2. Work with models that are bigger (more compute resources, higher latency)
3. Produce open-ended output (harder to evaluate)

AI engineering is less about model development and more about adapting and evaluating models.

#### Product-First Development Paradigm
The availability of powerful foundation models has fundamentally shifted the development approach:
- **Traditional ML**: Start with data → Build model → Create product
- **AI Engineering**: Start with product → Use existing models → Collect data only if needed
- **Benefits**:
  - Faster time to market and validation
  - Lower initial investment
  - Ability to test product-market fit before model investment
- **When to invest in models**: Only after product shows promise and has user traction

### Decision Tree: Choosing Your AI Approach

```
Start: Do you need AI for your application?
│
├─ No → Use traditional software engineering
│
└─ Yes → Can you achieve 80% quality with prompting alone?
    │
    ├─ Yes → Start with prompt engineering
    │        └─ Need domain knowledge? → Add RAG
    │
    └─ No → Do you have labelled training data?
        │
        ├─ No → Collect data or use synthetic generation
        │
        └─ Yes (1000+ examples) → Consider fine-tuning
            │
            └─ Still not sufficient? → Evaluate building custom model
```

Model adaptation can be divided into two categories depending on whether they require updating model weights:
- Prompt-based technique
  - model without updating the model weigths
  - adapt a model by giving instructions and context
  - easier to get started and requires less data
  - not enough for complex task or applications with strict performance requirements
- Finetuning
  - requires updating model weights
  - more complicated and require more data
  - improve model quality latency and cost significantly

#### Model Development
Model development is associated with traditional ML enginerring.
Main three responsibilities: modelling, training, dataset engineering, and inference optimisation.

#### Modelling and Training
Refers to the process of designing a model architecture, training it, and finetuning it.
ML knowledge is valuable as it expands the set of tools that you can use and helps troubleshooting when model doesn't work as expected.

- Quantization
  - process of reducing the precision of model weight
  - changes the model weight value but isn't considered training
- Pre-training
  - Training model from scratch
  - most resource intensive
- finetuning
  - Continuing to train a previous trained model
  - done by application developer
- Post training
  - process of training model afer the pre-training phase
  - done by model devloper
- Dataset engineering
  - curating, generating and annotating the data needed for training and adapting AI models
  - traditional ML engineering mostly close-ended
  - Foundation model are open-ended
  - AI engineering: deduplication, tokenisation, context retrieval and quality control
  - Expertise in data is useful when examining a model.
    - evaluate strenght and weakness of model
- Inference optimisation
  - Making models faster
  - Foundation model challenge: autoregressive: tokens are generated sequentially
- Application development
  - Foundation model applications should make differentiation gained through the application development process
  - application development layer consist of responsibilities:
    - evaluation
    - prompt engineering
    - AI interface
- Evaluation
  - Mitigating risks and uncovering opportunities
  - evaluation is needed to select models, to benchmark progress, to determine whether an application is ready for deployment and detect issues and opportunities for improvement in production.
  - **AI-as-a-Judge approach**: Using AI models to evaluate other AI models' outputs
    - More scalable than human evaluation
    - Can assess subjective qualities (helpfulness, coherence, safety)
    - Requires careful calibration against human judgement
    - Best used alongside traditional metrics and human review
- prompt engineering and context construction
  - Prompt engineering is about getting AI models to express the desirable behaviour from the input alone without changing the model weights
  - may need to provide the model with a memory management system
- AI interface
  - Creating an interface for end users to interact with your AI applications.
  - Provide interfaces for standalone AI applications or make it easy to integrate AI into existing products
    - Standalone
    - Browser extension
    - Chatbot
  - Chat interface is most commonly used, can be voice-based 

#### AI Engineering vs Full Stack Engineering
AI engineering closer to full stack development. Full stack developer is able to quickly turn ideas into demos, get feedback and iterate.
With AI models readily available, it's possible to start with building the product first, and only invest in data and models once the product shows promise.

> **Key Takeaways: AI Stack**
> - Three layers: Application Development, Model Development, Infrastructure
> - AI engineering focuses more on adaptation and evaluation than model development
> - Product-first approach: build applications using existing models before investing in custom models
> - AI engineers are closer to full-stack developers than traditional ML engineers

## Summary

This chapter establishes AI Engineering as a distinct discipline emerging from the democratisation of foundation models. Key insights include:

1. **Foundation Models Evolution**: From language models to multimodal systems through self-supervision
2. **Paradigm Shift**: Product-first development approach vs traditional ML's model-first approach
3. **Eight Major Use Cases**: Ranging from low-risk internal tools to high-risk customer-facing applications
4. **The 80/20 Challenge**: Reaching production quality requires disproportionate effort in the final 20%
5. **Three-Layer Stack**: Application, model, and infrastructure layers with distinct responsibilities

The transition from ML to AI engineering represents a fundamental shift in how we build intelligent applications, prioritising rapid iteration and existing model adaptation over custom model development.

## Glossary

**AI Engineering**: The discipline of building applications using pre-trained foundation models rather than developing models from scratch

**AI-as-a-Judge**: Using AI models to evaluate outputs from other AI models, enabling scalable quality assessment

**Autoregressive Model**: Language model that predicts the next token based only on preceding tokens

**CLIP**: Contrastive Language-Image Pre-training; pioneering multimodal model connecting vision and language

**Context Window**: Maximum number of tokens a model can process in a single request

**Crawl-Walk-Run Framework**: Microsoft's approach to gradually increasing AI automation in production

**Embedding**: Vector representation capturing semantic meaning of text, images, or other data

**Fine-tuning**: Continuing to train a pre-trained model on task-specific data

**Foundation Model**: Large-scale AI model trained on broad data that can be adapted for various tasks

**HITL (Human-in-the-Loop)**: System design involving human oversight in AI decision-making

**Last Mile Problem**: The disproportionate effort required to improve AI quality from 80% to production-ready

**LLM (Large Language Model)**: Foundation model specifically trained on text data

**LMM (Large Multimodal Model)**: Foundation model capable of processing multiple data modalities

**Parameter**: Learnable variable in a neural network; model size measured in parameter count

**Prompt Engineering**: Crafting inputs to elicit desired behaviour from AI models without changing weights

**Quantisation**: Reducing precision of model weights to decrease size and increase speed

**RAG (Retrieval Augmented Generation)**: Enhancing model responses with retrieved external knowledge

**Self-Supervision**: Training approach where labels are derived from the input data itself

**Token**: Basic unit of text processing in language models (roughly 3/4 of a word)

**Tokenisation**: Process of breaking text into tokens for model processing

**TPOT (Time Per Output Token)**: Latency metric measuring generation speed

**TTFT (Time To First Token)**: Latency metric measuring initial response time

