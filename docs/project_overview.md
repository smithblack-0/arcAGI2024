# Towards Developing a Recurrent Neural Network Exhibiting Online Neuroplasticity through Shuffled Vocabulary Pretraining

## Abstract

Current language models, primarily built on transformer architectures,
excel in a variety of natural language processing (NLP) tasks but suffer from
inherent limitations in scalability and adaptability. These models require
extensive computational resources for periodic retraining to incorporate new
information and are typically constrained to processing sequences of fewer than
10,000 tokens. This restricts their ability to maintain long-term context and
adapt continuously without explicit retraining. 

This proposal introduces a novel recurrent neural network (RNN) architecture designed 
to enable continuous learning through a unique pretraining methodology termed the 
"shuffled vocabulary pretraining task." This approach allows the model to autonomously learn and adapt
in real-time by consuming domain-specific texts without the need for additional
fine-tuning. The shuffled vocabulary pretraining task involves randomly
reassigning tokens within specific categories and providing minimal initial
translations, compelling the model to dynamically reorganize its understanding
and build a robust knowledge base on-the-fly.

Additionally, the proposed model incorporates a reversible architecture that
facilitates efficient training over extended sequence lengths by reconstructing
activations during backpropagation, thereby significantly reducing memory
consumption. Preliminary technical developments have successfully demonstrated
the model's capability to train over long token lengths of up to 30,000 tokens
using GPU accelerators. The next phase of this research aims to validate the
model's continuous learning and adaptability through a series of evaluation
metrics, including one-shot language learning, model neuroplasticity, and
performance on advanced benchmarks like SQuAD Hard and BLEU scores.

By addressing the static nature of traditional transformers, this project aspires
to advance towards more dynamic and adaptable AI systems capable of continuous
learning and real-time adaptation. The successful implementation of this
architecture could significantly enhance applications ranging from real-time
language translation to domain-specific expertise development, contributing
meaningfully to the pursuit of artificial general intelligence (AGI).

## Introduction

## Literature review

TO DO. Among other things, cover

* Current NLP state of the art is transformers, but they have bad behavior when being trained over long 
  sequences in terms of memory and computation time
* Recurrent networks like LSTM have a smaller footprint, and reversable architectures/checkpointing
  can make this even smaller
* Recurrent networks tend to have issues with exploding/vanishing gradients, though correct gate
  design can assist with that
* Recurrent networks tend to have issues with content that exceeds the sequence length it was trained over
* Linear transformers can be reformatted as a recurrent network process. 
* Reversable networks allow you to recompute activations when needed, greatly assisting
## Research Questions and Hypotheses

### Research Questions

This proposal builds upon foundational work where significant aspects of the
specialized architecture have already been demonstrated. The research
questions are categorized into **Completed Research Questions** and
**Current Research Questions** to reflect the progress made and the areas
that remain to be investigated.

**Completed Research Questions:**

1. **RQ1:** Can a recurrent network be redesigned to maintain only a
    segment of the sequence in memory at a time without compromising
    processing efficiency?  
    *(Demonstrated)*

2. **RQ2:** Can a reversible recurrent architecture with wide batch sizes
    achieve acceptable token processing rates over extended sequence
    lengths (e.g., 30,000 tokens)?  
    *(Demonstrated)*

3. **RQ3:** Do wide batch sizes mitigate the lack of parallelization
    inherent in recurrent architectures, thereby enhancing training
    efficiency to practical levels?  
    *(Demonstrated)*

**Current Research Questions:**

4. **RQ4:** Can an explicit loss function based on a gradient length
    proxy effectively prevent vanishing gradients in recurrent networks?

5. **RQ5:** Does tuning the model to maintain long-term state through a
    "gated average" mechanism, combined with a novel pretraining task,
    improve the recurrent architecture's ability to generalize beyond
    existing sequence length limitations?

6. **RQ6:** Does a model with parameter-based embeddings -in contrast to mutable
   embeddings that can be modified on the fly by the model - still provide sufficient
   neuroplasticity to meet the benchmarks?

### Hypotheses

Based on the completed and current research questions, the following
hypotheses are proposed to predict the outcomes of implementing advanced
training techniques and pretraining tasks:

1. **H1:** Recurrent networks are just as capable as transformer-based
    models, particularly when extra care is taken during training to
    maintain explicit gradient pathways.
    *(Tested via performance metrics on standard NLP benchmarks compared to other long sequence models)*

2. **H2:** The 'Causally Linked Articles' pretraining task is sufficient to
    encourage the recurrent model to begin building a knowledge database in its memory
    system to maintain important information based on what is being read.
    *(Tested via performance metrics on 'Mass Context Question Answering' tasks)*

3. **H3:** The 'Shuffled Vocabulary' pretraining task is sufficient to
    convince a recurrent model to look for and adapt to patterns such that
    it exhibits neuroplasticity—it will begin to mimic the content it is
    forced to read.
    *(Tested on 'Template Matching' benchmark and 'One-Shot Language' online learning tasks)*

4. **H4**: The model can be primed, after completing a knowledge base ingestion, to convert
           between domains in a completely online process with no additional fine tuning. 
    *(Tested on BLEU. Any score above random chance indicates success, as the model 
      did not undergo gradient descent in the new one-shot language.)*

## Objectives

The primary objective of this project is to develop and validate a recurrent
neural network architecture designed from scratch to support continous learning and 
exhibit neuroplasticity. Specifically, we aim to:

**Primary Objectives:**

- **Demonstrate the feasibility of recurrent networks** as a transformer substitute by
  specially designing the network and losses to avoid many common pitfalls and over
  extremely long training durations.
- **Demonstrate model neuroplasticity** by having the model learn to automatically 
  adjust generative patterns to match templates like a [context] ... [question]... [answer] model
  pattern simply by reading a bunch of matching cases.
- **Demonstrate one-shot learning** of a new language (German) by providing
  minimal initial translations and exposing the model to extensive German text.
- **Validate continuous adaptation** by having the model improve its
  understanding over time without explicit retraining.
- **Establish the viability** of this approach for artificial general
  intelligence (AGI) tasks, setting the foundation for future exploration in
  domain-specific online learning.


## Methodology

### Pretraining Pipeline

#### Traditional Pretraining Task

The initial phase of training will utilize a "traditional" pretraining dataset 
to establish a foundational understanding of language. This dataset, such as 
WikiText (English) or a comparable high-quality corpus, is characterized by 
its consistent style, single-language focus, and suitability for long-context 
training. The model will process sequences up to 32,000 tokens in length, 
leveraging the efficiency of the recurrent architecture to handle extended 
contexts.

During the technical demonstrator phase, the model achieved a processing rate 
of approximately 1,000 tokens per GPU per second on an L4 GPU, using unusually 
wide batch sizes of 800. This configuration was necessary to ensure that the 
GPU remained saturated, given the non-parallelizable nature of recurrent 
networks. Significant optimizations since the demonstrator phase, including 
improvements in memory logic and the integration of TorchScript, are expected 
to substantially increase performance, although further empirical testing is 
required to confirm exact figures.

To optimize computational efficiency, articles of similar length will be 
grouped together during training to minimize padding overhead. Next-word 
prediction will be used as the loss function in this phase, aligning with the 
generative nature of the model. This task emphasizes the acquisition of core 
linguistic patterns, including grammar and within-sequence predictive 
relationships, providing a robust foundation for the specialized pretraining 
tasks that follow.

#### Causally Linked Articles Pretraining Task

The "Causally Linked Articles" pretraining task is a cornerstone of this project, 
designed to teach the model to construct and utilize a long-term context bank. By 
presenting sequences of articles linked by causal relationships, this task 
encourages the model to actively accumulate and manage knowledge across extended 
contexts.

1. **Purpose and Dataset Characteristics**:  
   - The task focuses on processing sequences of causally linked articles to 
     simulate scenarios where retaining prior information improves understanding 
     of subsequent content.  
   - Academic papers are an ideal starting point due to their structured citation 
     relationships, but the methodology is adaptable to any dataset with causal 
     links, such as legal documents or linked news articles.  
   - The dataset is constructed by concatenating articles based on their causal 
     relationships, emphasizing the necessity of remembering earlier articles to 
     interpret later ones. Sequences can extend to lengths of 64,000 tokens or 
     more.

2. **Training Strategy**:  
   - Articles are processed into sequences based on their citation relationships. 
     Starting with a root article, citations are followed to form a tree 
     structure.  
   - This tree is traversed in breadth-first order to create concatenated 
     sequences of articles.  
   - Wide batch sizes (e.g., 800 articles per batch) are used to saturate the 
     GPU, enabling efficient processing despite the recurrent architecture’s 
     inherently sequential nature.

3. **Encouraging Knowledge Compression**:  
   - The number of articles concatenated in a sequence ensures that the model 
     cannot simply memorize all the information. Instead, it must compress the 
     content into an internal knowledge base to achieve acceptable performance.  
   - This design compels the model to prioritize efficient knowledge encoding, 
     storage, and retrieval, simulating real-world scenarios where long-term 
     memory and generalization are crucial.  
   - This knowledge compression strategy directly addresses RQ5 by enabling the 
     model to generalize beyond its training sequence length. By shifting from 
     memorization to knowledge accumulation, the model is expected to perform 
     effectively over even longer contexts, improving its generality.

This task directly supports the project’s goal of fostering adaptive learning by 
training the model to build and utilize a dynamic knowledge base. By confronting 
the model with extended sequences of causally linked content, this pretraining 
task creates the conditions necessary for the emergence of long-term memory 
utilization and enhanced generality.

#### Shuffled Vocabulary Pretraining Task

The shuffled vocabulary pretraining task is a key innovation in this project, 
designed to instill neuroplasticity in the model while encouraging it to build 
understanding through cross-referencing known and unknown relationships. By 
disrupting token-level assumptions and forcing the model to infer meaning from 
context, this task enhances both adaptability and the capacity for intelligent 
generalization.

1. **Dataset Selection**:  
   - The dataset for this task will be drawn from a holdout subset of WikiText 
     or another comparable dataset. This ensures the text content is of high 
     quality, diverse, and linguistically challenging, while preventing overlap 
     with earlier pretraining stages.  
   - The use of a holdout subset ensures that the model approaches the task with 
     no prior token associations from earlier training, preserving the integrity 
     of the shuffled vocabulary objective.

2. **Vocabulary Shuffling**:  
   - Tokens are randomly reassigned within specific categories, such as stems or 
     subwords. Additionally, a subset of tokens undergoes fusion (combining 
     tokens into larger units) or fission (splitting tokens into smaller units) 
     to further disrupt straightforward token mappings.  
   - Fission and fusion are crucial for preventing the model from relying on 
     frequency analysis to deduce token meanings, ensuring it must rely on 
     context and relational inference instead.

3. **Initial Translations**:  
   - The model is provided with translations for 2,000 of the most common words, 
     selected from the top 10,000 tokens. These "Rosetta Stone" pieces are chosen 
     to include relatively "easy" or high-frequency words that form a strong 
     starting point for building comprehension.  
   - This initial translation provides the model with an anchor for known 
     relationships, allowing it to cross-reference and infer the meanings of 
     unknown tokens through reading.

4. **Exposure to the Permuted Corpus**:  
   - The model processes text encoded with the shuffled vocabulary, spanning 
     sequences of 100,000 tokens or greater.  
   - This extensive exposure requires the model to actively adapt its internal 
     hypotheses about token meanings and to build relationships between known 
     and unknown tokens dynamically.

5. **Gradual Loss Scaling**:  
   - To allow the model time to adapt, loss importance is gradually scaled up 
     over the first 5,000 tokens within each training sequence.  
   - This scaling continues as the model processes extended sequences, ensuring 
     it has sufficient time to refine its internal hypotheses about token 
     meanings.

This task serves multiple purposes. First, it fosters neuroplasticity by 
requiring the model to constantly update its understanding of token meanings. 
Second, it trains the model to build comprehension by cross-referencing known 
tokens with unknown ones, mimicking the way humans learn new languages or 
concepts. Finally, the combination of vocabulary shuffling, fission, and fusion 
ensures that the model cannot rely on simple statistical shortcuts like 
frequency analysis, pushing it toward deeper, contextual understanding.

The model's "weighted average" of linear attention states is specifically 
designed to support this kind of task, enabling it to track shifting token 
meanings over extended sequences. By fostering both adaptability and 
intelligence, the shuffled vocabulary pretraining task prepares the model for 
online learning, domain transfer, and generalization to novel contexts.

#### Conclusion: Building a Foundation for Neuroplasticity and AGI-Like Behavior

The pretraining pipeline is designed as a progression, with each task building 
upon the previous to create a model capable of advanced generalization, memory 
utilization, and neuroplasticity. This sequence not only equips the model to 
understand language but also primes it for dynamic, adaptive reasoning.

1. **The Traditional Pretraining Task** lays the foundation by teaching the 
   model core linguistic principles. Through exposure to high-quality language 
   data, the model develops a fundamental understanding of grammar, syntax, and 
   within-sequence relationships, establishing the building blocks for language 
   comprehension.

2. **The Causally Linked Articles Pretraining Task** shifts the focus to long-
   term knowledge accumulation. By presenting sequences of causally related 
   content, this task encourages the model to synthesize large volumes of 
   information into a coherent internal knowledge base. This primes the model 
   to manage extended contexts effectively and fosters an intuition for 
   gathering and storing meaningful relational understandings over long 
   durations.

3. **The Shuffled Vocabulary Pretraining Task** builds on this primed 
   foundation, challenging the model to transition from passive knowledge 
   accumulation to active reasoning. By starting with a limited set of axioms 
   (initial translations) and forcing the model to infer relationships between 
   known and unknown tokens, this task develops the ability to adapt dynamically 
   to new contexts. It teaches the model to distrust static representations and 
   instead prioritize continuous hypothesis testing and pattern discovery. This 
   process endows the model with neuroplasticity, enabling it to generalize 
   beyond its training and mimic the adaptive behaviors characteristic of AGI.

Together, these tasks form a robust pipeline that transforms a basic language 
model into a system capable of adaptive learning, long-term memory utilization, 
and research-like behavior. By carefully sequencing these pretraining steps, 
the pipeline sets the stage for a model that can move beyond static predictions, 
actively constructing knowledge and reasoning through novel challenges. This 
progression directly supports the project's overarching goal of advancing 
towards artificial general intelligence.

### Evaluation Metrics and Connections to Hypothesis.

#### Traditional Pretraining Perplexity

- After or even during the traditional pretraining task, we can monitor the perplexity
  to get a feel of how the model is converging. We use the test split for this, allowing
  comparison to other models.
- Compare it on the Long Range Area benchmark task.

### Evaluation Metrics

- **Model Neuroplasticity**: Verify that the pretraining task in fact allows neuroplasticity,
  by providing text that shows a template with control tokens be used. For example, only show
  conversations with a [USER] and [AI] tag, and see if the model will start replying as the AI.
  Or language drift - if suddenly all [no] tokens must be preceded by [bacon], does the model 
  follow along?
- **SQUAD Hard**: Rather than being given a block of context, then questions, the
  model is given the entire concatenated set of context to learn and then asked
  questions on it. This means the model must maintain its own knowledge base. 
- **One-shot German learning**: Measuring the model's perplexity on German text to
  assess language acquisition. This will include an initial translation like in
  the shuffled task, around 200,000 german tokens, then perplexity over unreviewed
  text samples.
- **BLEU**: No language task would be complete without BLEU. Take the previous model,
  apply text-to-text techniques with [SEP] tokens, and see if the model will
  start translating from english to german.



### Recurrent Reversible Network Architecture

This is a highly nonstandard training mechanism. Unusually, the targeted training
tasks:

- Training occurs over enormously long context windows like 32,000 tokens.
- Inference is expected to occur over even longer distances, as are metric
  mechanism.

This presents significant issues for usage with the standard transformer mechanism.
As a result, a nonstandard model is needed.

Our model uses a reversible architecture that allows us to reconstruct activations during
backpropagation without storing all intermediate states, significantly reducing memory consumption.
Additionally, it uses a specialized weighted average technique along with special pretraining tasks
to allow the model to learn online just by reading.

#### Motivations for a Recurrent Architecture

It is worth having a brief discussion on why we decided on returning to a 
recurrent architecture after the academic communities general movement away. 
Recurrent architectures tend to perform poorly, so we should at least discuss
why this one may perform well.

**Pros and Cons**

The academic community originally moved away from the usage of a recurrent architecture
due to several primary issues. These included, but were not exclusively limited to, 

- Vanishing and exploding gradients. (source)
- Poor performance over sequence lengths longer than the training length (source)
- Generally, worse performance than transformers.
- Issues with saturating computational devices. Transformers get around this by 
  parallelization, but recurrent models can only really step forward one token at a time.

However, some attractive features they did possess was

- O(N) training time memory requirements
- Constant inference time memory requirement
- O(N) inference time computation time requirements

These compare favorably to transformers in their respective
attributes. Transformers have:

- O(N^2) training requirements

**Managing the cons**

The architecture and training tasks used in this model is designed explicitly to manage
the cons. 

- A specialized loss exists based on a distance metric that directly penalizes the model if 
  gradient pathways are not distributed between pathways to early timesteps, moderate timesteps,
  and late timesteps from the current timestep position. This is to control vanishing gradients
- Careful construction of the recurrent mechanism and losses help control exploding gradients.
- The recurrent mechanism is a restructured version of linear attention,
- The training tasks are specifically designed to bias the model with the expectation that 
  keeping long term context around is useful 
- Extremely wide batch widths (800) is used to ensure the accelerant is fully saturated and 
  maintain decent token lengths.


#### Reversibility and Memory

The reversible architecture is very carefully designed to keep the computation
devices saturated with while avoiding running out of memory. It does this using 
a checkpointing system. 

In particular, it is the case that each token is processed one at a time, in an 
extremely wide batch, with the processing of the token consisting of a checkpoint
location. Memory is carefully managed to ensure that the size of the graph that 
must be maintained is restricted to solely enough to hold a single token's computation
graph at a time. This greatly reduces the amount of memory needed. Loss is taken
while maintaining this graph, rather than in one large step at the end, in such
a manner that the math is theoretically equivalent.

The model is also reversible, again to save memory. Three operations occur. A forward
mode that produces the activations. A reverse pass recovers these activations while 
keeping the gradients around. Then a backwards pass actually collects the loss.

A special trainer class controls this to ensure it behaves like a standard next 
token prediction loss process would while avoiding running out of memory.

#### Architecture implementation

The implementation itself is a set of transformer-like recurrent layers 
in which each has a feedforward, and a memory process. Current plans are
to use a linear attention analogue in the memory process, and try a few
others as well. 

The recurrent process has gates that determine how much of each memory
update is committed or erased from memory. These gates are defined only
from the layer inputs, not the recurrent state, unlike in an LSTM. This
makes the process much easier to reverse, but perhaps a little less 
stable - the model WILL drift in output over time.

#### Technical status

The architecture is almost completely coded, and a proof of concept
has been run using it. We did manage to train on wikitext2. What remains 
of this is to rebuild the training class for distributed work.



## Expected Outcomes

- **Technical demonstration** (Completed): Can train over arbitrary token lengths, without issue 
    at an acceptable token rate.
- **Proof of Concept**: 
  - Validate that the model can be trained to something matching existing models with similar 
    parameter sizes.
  - Perform vocabulary shuffle learning tasks.
  - Run the neural-plasticity demonstration task. Show the model will begin to modify 
    its control token pattern just by reading.
  - Run the SQUAD hard metric.
- **Full Scale Demonstration**:
  - Train the foundation model for real.
  - Apply the SQUAD hard metric.
  - Apply the One-Shot Language metric
  - Apply the BLEU test.

## Timeline and Milestones

Note: Timeline is quite aggressive. This is because the majority of the model has already
been coded and a technology demonstration has already been performed.

- **Month 1**:
  - Rebuild trainer class for compatibility distributed training.
  - Develop and flesh out support for distributed training including logging, exceptions, and 
    dead node recovery.
  - Secure distributed training resources.
  - Implement Pattern Neuroplasticity metric task.
- **Month 2**:
  - Run initial pretraining case, without shuffled vocabulary. (background)
  - Program hyperparameter search code.
  - Implement the Academic Biasing pretraining task. 
  - Implement the shuffled vocabulary pretraining task. Reuse most of existing 
    pipeline mechanism.
  - Implement the SQUAD hard and One-Shot German pretraining task. 
  - Test pretrained foundation is comparable to peer models with similar 
    number of parameters. Correct issues. Minor performance loss is okay.
- **Month 3**:
  - Run pretraining and shuffling for real, with hyperparameters
  - End up with foundation model.
  - Apply metrics.
  - Buffer time for unexpected issues.


## Budget Breakdown

- **Prototyping Budget ($2,000)**:
  - Compute time on clusters for initial runs.
  - Up to 20 pretraining runs at up to $100 per run.

- **Personnel ($5,000)**:
  - Three months' salary for one novice researcher working nearly full-time.

- **Training Budget ($12,000)**:
  - Main training run, including short hyperparameter search.

**Total Budget: $18,000**

## Conclusion

This proposal outlines a plan to develop a recurrent neural network architecture
capable of continuous learning through the shuffled vocabulary pretraining task.
By successfully demonstrating one-shot learning of a new language and continuous
adaptation, this project aims to overcome the limitations of traditional
transformer models and contribute significantly to the field of artificial
general intelligence.

---

*For further discussions or inquiries about this proposal, please contact Chris O'Quinn at
chrisoquinn.2@gmail.com
