# Introduction to Building AI Applications with Foundation Models

AI in one word post-2020: scale. The scaled up AI models produced powerful and capable, but not affordable for independant model. As a result, few organisation can afford, which lead to the emergence of **model as a service**. 

AI engineering: the process of building applications on top of the model as a service from the organisation.

This capter can be summarised in three chapters:
1. Overview of Foundation models
   1. Covers why we want to use AI Engineering
2. Foundation model use cases
   1. Covers what can be done with AI Engineering
3. Planning AI Applications
   1. Covers how to do AI Engineering

## Overview of Foundation models 

### History

Foundation models emerged from large language model.

LLM has only been able to scale because of the self-supervision.

A language model encodes statistical information about language. How likely a word is to appear in a given context.
- i.e. "My fvorite colour is ___": blue over car

The statistical nature of language was discovered long time ago:
- Sherlock Holmes
- Claude Shannon, "Prediction and Entropy of Printed English"

### Token

- Token:
  - The basic unit of language model.
  - A token can be a character, a word, or part of a word (i.e. -tion)
- Tokenization:
  - The process of breaking the original text into token
  - Average token is approximately 3/4 the length of a word
- Model's vocabulary:
  - The set of all tokens a model can work with.

Why LLM uses token as their unit over words or characters?
1. More meaning ful to model
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
- Supervision: trained using labeled data.
  - Expensive and time consuming
- self-supervision: trained without labeleld data.
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

### From LLM to Foundation MOdels

Tranditional model
- Natural language processsing (NLP) deals only with text
- Computer vision deals only with vision
- Audio model can handle speech recognition and synthesis

Multimodal model:
- Foundation model
- A model that can work with more than one data modality
- A generative mododal model is called large multimodal model (LMM)
- Multimodal model generates the next token conditioned on both text and image token

I was working on computer vision, but it can be replaced by multimodal model.

CLIP:
- language model OpenAI trained using (image, text) pairs that co-occurred on the internet.
- First model generalize to multiple image classification tasks.
- CLIP is an embedding model

Embedding: Vector that aim to capture the meaning of the original data.

3 Techniques to generate what you want with the model
- Prompt engineering: Craft instructions with example
- Retrieval augmented generation (RAG): Using a database to supplement the instruction
- Finetune: further train the model on dataset of high-quality product description
- Considerations:
  - How much data is needed depends on what technique you use
  - Building own model vs Use existing model <-> Buy or build question in traditional business

### From Foundation model to AI engineering

AI engineering: Process of building applications on top of foundation model
- ML Engineering: Developing ML models
- AI Engineering: Leverages existing models.
- Three factors
  1. General purpose AI capabilities: 
     1. Model can do existing task better and more
     2. Help train even more powerful model in the future (self improving?)
  2. Increased AI investment
     1. AI applications become cheaper to build and faster to productise, which is attractive to invester.
  3. Low entrance barrier to build AI applications
     1. Easier to leverage AI to build applications
     2. Build applications with minimal coding


## Foundation model use cases

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

Use canses can be analysed in two categories: enterprise and consumer applications.
- i.e. Image and video production
  - consumer: photo and video editing, designer
  - enterprise: presentation, AD generation


Cateogry|Examples of consumer use cases|Examples of enterprise use cases
-|-|-
Coding|Coding|Coding
Image and video production|Photo and video editing<br/>Designer|Presentation<br/>AD generation
Writing|Email<br/>Social media and blog posts|Copywriting, search engine optimisation (SEO)<br/>Reports, memos, design docs
Education|Tutoriing<br/>Essay grading|Employee onboardinig<br/>Employee upskill training
Conversation bots|General chatbot<br/>AI companion|Customer support<br/>Product copilots
Information aggregation|Summarisation<br/>Talk to your docs|Summarisation<br/>Market research
Data organisation|Image serach<br/>Memex|Knowlege management<br/>Document processing
workflow automation|Travel planning<br/>Event planning|Data extraction, entry, and annotations<br/>Lead generation

Important to note that an application can belong to more than one category.
- enterpreise prefer applications with lower risk
  - i.e. deploy internal facing application (text summarisation, knowledge management)
  - customer facing higher risk (External chatbox, recommendation algorithm)

### Coding

Coding is the most popular use case at the moment
- Documentation
- Code generation
- Code refactoring

AI can certainly make engineers more productive, and can disrupt the outsourcing industry.

### Image and video production

AI is great for creative tasks.
Successful startups:
- Midjourney for image generation
- Adobe Firefly for photo editing
- Runway, Pika Labs and Sora for vieo generation

For enterprises ads and marketing have incorporated AI.
- Generate promotional images and video
- Brainstorm ideas or generate first draft for human experts

### Writing

Autocorrect and auto-completion powered by AI have been used for long.
This is quite tedious, and human has high tolerance for mistakes.

ChatGPT helps close the gap in output quality between workers. Helpful for those with less inclination for writing.
- Consumers: help to communicate better such as email
  - Writers use AI to write books
  - AI help to imporve their writing in note taking
- Enterprise: sails, marketing and general team communication.
- SEO

### Education
- Help to learn faster
- Personaliszed lecture plan
- language learning
  - roleplay, generate quizzes, debate partner
- tutor to learn any skill

### Conversational Bots
- Customer support bots
- Filing insurance claim, texes, corporate policies
- Smart NPC

### Information Aggregation
- Process documents; contracts, disclosures, papers
- talk-to-your-docs
- Summarise websites, research and create report

### Data Organisation
- Essential to organise all dtaa in a way that can be searched later
- Data analysis

### Workflow Automation
- Ultimately, AI should automate as much as possible
- Lead management, invoicing reimbursements, managing customer requests, data entry
- Agents: AI that can plan and use tools

## AI Application

### Planning AI Applications
- Building application is one of the best ways to learn
- If you are doing it for a libving
  - Take step back
  - Consider: why you are building this, how you should go about it.
  - Hard to create a profitable product, easy to demo with foudnation models

#### Use Case Evaluation
Example of differernt levels of of risk
1. If you don't do this, competitors with AI can make you obsolete.
2. If you don't do this, you'll miss opportunities to boost profits
3. You're unsure shere AI will fit into your business yet, but you dont' want to be left behind.

### The role of AI and human in the application
Role of AI in AI product:
- Critical or complementary
  - If app can still work without AI, AI is complementray to the app
  - More critical AI requires more accurate and reliable the AI part has to be.
- Reactive or proactive
  - Reactive feature in reaction to users' requests
    - chatbot
    - latency can be matter
  - Proactive feature response when there's an opprtunity
    - traffic alert on google map is proactive
    - higher quality bar because user didn't ask for it
- Dynamic or static
  - Dynamic update continually with user feedback
    - Face ID
  - static updated periodically
    - object detection in Google Photo

Human-in-the-loop: Involving humans in AI's decision making processes

Microsoft's framework for gradually increasing AI automation in production: Crawl-Walk-Run:
1. Crawl: human involvement is mandatory
2. Walk: AI can directly interact with internal employee
3. Run: Increased automation, direct AI interactions with external users

### AI product defensibility
Low entry barrier is blessing and a curse.
- Easy for you to build, it's easy for your competitor to build.

Three types of competitive advantages:
- technology
- data
  - If startup can get to market first and gather sufficient usage data
  - The data can be used to guide the data collection and training process
- distribution

### Setting Expecations
Figure out what success looks like: how will you measure succes.
Chatbot example
- What percentage of customer message chatbot automated
- How manu more message chatbot allow you to process
- How quicker can you respond using the chatbot
- How much human labor can chatbot save you?
It's important to track customer satisfaction and feedback in general.

Usefulness thresholds include following metric groups:
- Quality metrics to measure quality of the chatbot's reponses
- Latency metrics: TTFT (time to first token), TPOT (time per output token) and total latency
- Cost meric: how much it costs per inference requesets
- Interpretability and fairness

### Milestone planning
Evaluate existing models to understand their capabilitities.
Goals will change after evailuation.
Planning an AI product needs to account for its last miles challenge.

The journey from 0 to 60 is easy, whereas progressing from 60 to 100 becomes exceedingly challenging.

### Maintenance
Need to think about how the product might change over time, how it should be maintained.
Model inference, the proces of computing an output given an input, is getting faster and cheaper.
Run a cost benefit analysis. It's getting easier to swap one model APi for another.


## AI stack
Important to recognise that AI engineering evolved out of ML engineering.
Position AI engineers and ML engineers, their roles have significant overlap.

### Three layers of the AI stack

Stacks:
1. applciation development
   1. anyone can use model to develop application
   2. requires rigorous evaluation
   3. good interfaces
2. model development
   1. tooling for developing models
3. infrastructure
   1. tooling for model serving
   2. manging data
   3. compute and monitoring

### AI Engineering vs ML Engineering

important to understand how thigns have changed.
Buliding applciation using foudnation model today differs from traditional ML engineer.
1. Use model someone else has trained for you
2. Work with models that are bigger
   1. more compute resources, higher latency
3. Produce open-ended output
   1. harder to evaluate

Ai engineering is less about model devleopment and more about adapting and evaluating models.
Model adaptation can be divided into two categories depending on whether they require updating model weights:
- Prompt-based technique
  - model without updating the model weigths
  - adapt a model by giving instructions and context
  - easier to get started and requiers less data
  - not enough for complex task or applications with strict performance requirements
- Finetuning
  - requires updating model weights
  - more complicated and require more data
  - improve model quality latency and cost significantly

#### Model development
Model development is associated with traditional ML enginerring.
Main three responsibility, modeling, training, dataset enginerring and inference optimisation.

#### Modeling and training
MRefers to the process of comping up with a model architecture, training it and fintuning it.
ML knowledge is valuable as it expands the set of tools that you can use and helps troubleshooting when model doesn't work as expected.

- Quantization
  - process of reducing the precision of model weight
  - changes the model weight value but isn't considered training
- Pre-training
  - Training model from scratch
  - most ressource intensive
- finetuning
  - Continuing to train a previous trained model
  - done by application developer
- Post training
  - process of training model afer the pre-training phase
  - done by model devloper
- Dataset engineering
  - curating, gereating and annotating the data needed fro training and adapting AI models
  - traditional ML engineering mostly close-ended
  - Foundation model are open-ended
  - AI engineering: deduplication, tokenization, context retrieval and quality control
  - Experties in data is useful when examping a model.
    - evaluate strenght and weakness of model
- Inference optimisation
  - Making models faser
  - Foundation model challenge: autoregressive: tokens are generated sequentially
- Application development
  - Foundation model application should make differentiation gained through the application development proess
  - application development layer consist of responsibilities:
    - evaluation
    - prompt engineering
    - AI interface
- Evaluation
  - Mitigating risks and uncovering opportunities
  - evaluation is needed to select models, to benchmark progress, to determine whether an application si ready for deployment and detect issues and opportunities for improvenet in production.
- prompt engineering and context construction
  - Promt engineering is about getting AI modesl to express the desirable behaviour from the input alone without changing the model weight
  - may need to provide the model with a memory mangement system
- AI interface
  - Creating an interafce for end users to interact with your AI applications.
  - Provide interaces for standalone AI application or make it easy to integrate AI into existin product
    - Standalone
    - Browser extension
    - Chatbot
  - Chat interface is most commonly used, can be voice base, 

#### AI Engineering vs full stack engineering
AI engineering closer to full stack development. Full stack developer is able to quickly turn ideas into demos, get feedback and iterate.
With AI models readily available, it's possible to start wit building the product first, and only invest in data and modesl once the product shows promises.

## Summary

This chapter serve two purpose
- Explain the emergence of AI engineering as dscipline thanks to the foundation model
- Give an overview of process needed to build application on tiop of theses models.

This chapter covers evolution of AI.
- Transition from language models to large language models over self-supervision

This chapter discuss whther you should build AI application and considerations.

Finally AI engineering stack including how it changed from ML engineering

