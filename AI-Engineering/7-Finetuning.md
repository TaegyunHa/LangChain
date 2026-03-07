# Finetuning

## Benefits

- Domain specific capability
- Safty
- Instruction follownig ability
  - adhere output style and format
  - performance of tool calling, agent

## Overview

- Finetuning is one way to do transfer learning.
- Transfer the knowledge gained from one task to accelerate learning for a new related task
  - Base model: text completion
  - Finetune model: text-to=SQL
- Finetuning can extend the context length
  - long-context finetuning

> Transfer learning is widely used in computer vision.
> - Keep the low level weight
>   - Because this weight contribute to find a fundamental pattern such as line or curve
> - Training the high level weight
>   - This contribute to more complex pattern such as eye shape, ear shape, etc.

## When to finetune

### Benefit

- improve model's quality both general capacity and task specific capability
- bias mitigation
- imitate the behaviour of larger model
  - dilstillation

> It seems pretty large chunk of what tech companies do would be finetuning.

### Drawbacks

- Can degrade its performance for other tasks
- Resource intensive
  - Annotated data
  - Knowledge of how to train model
  - Policy and budget for monitoring, maintaining and updating the model
  
## Finetuning and RAG

- Use RAG when
  - model fails because it lacks information
    - model doesn't have the information
    - model has outdated information
  - for the `fact`
- Use finetuning when
  - model has behavioural issue
    - incorrect output
    - irrelavant to the task
    - incorrect output format
  - for the `form`

## Memory bottlenecks

Usually memory is the bottleneck when it comes finetuning. As a result, many technics were developed to overcome this.

- technics
  - quantization

> TensorRT compilation would do the quantisation?
> In media domain, this is important concept (pixel format)

## Finetuning technics

### Parameter efficient finetuning (FEFT)

- Adapter-based technic
  - add trainable parameters to the model
  - LoRA
- soft prompt based technic
  - introducing special trainable token
  - soft prompt
    - not human readable
    - typically continuous
    - trained during back propagation

> I was curious if we can add token for training. For example, type of conversation or expected output, etc. The soft prompt would be better approach as model will learn the pattern by itself.
> 
> I think LoRA is really claver. I never thought that full matrix can be expressed by two lower rank mattriecs, which will reduce the parameter significantly.

## Model merging

- delta vector
- pruning

> Is it possible to visualise the activated node and merge the model part based on the task? CNN has this approach.

## Other thought

> conventional programming use dll to optimise the programming. can this concept used for model if the model can be modularised?

## Glossary of Key Terms

- PEFT
  - Parameter efficient finetuning
  - adapter based technic
- Prompt caching
  - Repetitive prompt segment can be cached for reuse.
- trainable parameter
  - parameter that can be updated during finetuning
- loss
  - difference between the computed output and the expected output
- gradient
  - how much trainable parameter contributes to the mistake
- optimiser
  - how much each parameter should be readjusted
  - Adam, SGD