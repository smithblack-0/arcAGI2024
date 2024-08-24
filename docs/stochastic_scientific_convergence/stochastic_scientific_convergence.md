# Introduction

This document outlines a hybrid approach between reinforcement 
learning and supervised learning, designed to help a model both 
experiment to gather information and rapidly converge on an optimal 
outcome. The core objective is to generate a "rule," which is a 
sequence of actions (represented as embeddings) that can be executed 
by a mapping model to transform an input state into a desired output 
state. In this context, "experiments" are sequences of actions 
intended to maximize information gain, ultimately leading to the 
discovery of the optimal rule.

## Background

### Reinforcement Learning from Human Feedback (RLHF)

A common strategy for fine-tuning transformer models is Reinforcement 
Learning from Human Feedback (RLHF). In this method, a transformer 
generates multiple sequences in response to a prompt, and a human 
selects the best sequence. This selection is then used to guide 
teacher-forced training, fine-tuning the model's output distribution 
toward the most optimal direction.

### RLHF with Reward Models

A variant of RLHF involves replacing the human with an AI-based reward 
model. The reward model, trained by a human, provides feedback on the 
effectiveness of the generated sequences. This feedback is used to 
train the transformer, demonstrating the feasibility of automated 
feedback. The result is a target sequence that can be used for further 
training.

### Adaptive Computation Time (ACT)

Adaptive Computation Time (ACT) is a mechanism that allows 
computations to take a variable amount of time, combining intermediate 
results according to a halting probability. This flexibility is 
particularly useful when the complexity of the task varies, as it 
enables the model to allocate more resources to more challenging 
problems.

### ARC-AGI Challenge

The ARC-AGI (Abstraction and Reasoning Corpus - Artificial General 
Intelligence) challenge has proven to be extremely intractable for 
current techniques. It involves generalized reasoning in a manner that 
differs significantly from conventional training data. One of the 
major difficulties is the need to respect an underlying, but 
unspecified, Domain-Specific Language (DSL). This challenge 
underscores the limitations of current models and motivates the 
development of new approaches.

## IDEAS:

### Common Model Sequence

Basically, a sequence of embedidngs

* Keeps track of experiments that have been tried
* Seen by a number of different models
* Swap between models depending on what we are doing

### Subcase Proposer

The idea is the model decides what part of the problem to try
solving next itself.

It will generate based on the main problem a restricted subset of content
from the problem and apply reconstruction masks to some components. 

* Sees
  * Prompt
  * CMS 
* Produces
  * Subcase with Prompt and Solution.
  * Subcase prompt is built out of main prompt.
  * Options are include, mask, or drop.
* Loss/Targets
  * Largest perplexity given minimum masks.

### Optimal Context Proposer

* Sees:
  * Prompt Subcase
  * Solution Subcase
  * Common Model Sequence
* Needs to produce:
  * Context sequences that best help 



### Experiment 


### Rule Transformer

### Context Transformer

* Sees the unmasked examples
* Wants to provide the most informative set of context

### Reconstruction Transformer

* Sees some masked cases 
* Sees the context.
* Tries to reconstruct the masked case based on the context and the visible.

### Scientist Process

* Generate a set of context proposals
* Evaluate them in the reconstruction transformer
* Keep the most useful proposal.
* Repeat until reconstruction has mostly been successful.




## Objective Clarification

The primary goal of our model is to generate a "rule," which is a 
sequence of actions that can effectively operate the mapping function 
to transform an input state into a desired output state. The rule, 
composed of embeddings, is executed by the mapping model, which 
processes the input collection and the rule to produce the desired 
output collection. 

In the process of discovering this rule, the model 
conducts "experiments," which are sequences of actions intended to 
gather information about the scenario. The objective is to propose and 
evaluate these experiments to gain the most information, ultimately 
leading to the discovery of the optimal rule.

## Data format

We are assuming for the moment that the data being provided
consists of a collection of sequenced input and output mappings. 

During pretraining, with known example collections, some quantity
of output maps or input maps may have been replaced with a special
mask token, as in BERT. This specifies the model will need to figure
out the masked token given the context of the other examples.

During training for ARC-AGI, we can include the targeted output
state among the input examples and mask the output. This will then
ask the model to predict the missing output, just like during training.

## Mapper Model

The Reconstructor Model is a model 



## Model Structure

### Scientist Transformer

The "scientist" transformer is responsible for proposing experiments 
and selecting the most informative one based on the perplexity of the 
results. This transformer operates in a learning cycle, where it 
proposes a collection of experiments, runs them, and then selects the 
one that yields the highest information gain. The selected 
experiment's result is used as the starting point for the next 
iteration of the cycle.

### Rule Transformer

After each learning cycle, the model produces a halting probability 
and executes the "rule transformer" to generate a rule candidate. 
These rule candidates are then combined using the Adaptive Computation 
Time (ACT) mechanism to form the final rule. This rule is a sequence 
of embeddings designed to operate the mapping function and transform 
the input state into the desired output state.

### Mapping Model

The mapping model accepts the input collection and the rule produced 
by the rule transformer. It then applies the rule to the input 
collection to generate the output collection. The effectiveness of the 
rule in producing the correct output collection is a key measure of 
the model's success. The mapping model plays a crucial role during the 
experimentation phase, as running an experiment means feeding the rule 
and input collection into the mapping model to observe the resulting 
output.

### ACT Mechanism

The Adaptive Computation Time (ACT) mechanism allows the model to 
dynamically allocate more computation resources to more complex tasks. 
In this context, it is used to combine the rule candidates generated 
by the rule transformer, ensuring that the final rule is both accurate 
and efficient.

## Learning Cycle and Experimentation

### Learning Cycle

The learning cycle consists of the following steps:

1. **Propose Experiments:** The scientist transformer proposes a 
collection of experiments (sequences of actions).
2. **Run Experiments:** The proposed experiments are executed by 
feeding the rule and input collection into the mapping model, which 
generates the output collection for each experiment.
3. **Select the Best Experiment:** The model calculates the perplexity 
of each result and selects the one with the highest perplexity (most 
surprising). This result is then used as the starting point for the 
next cycle.
4. **Generate Rule Candidate:** After each cycle, the model produces a 
halting probability and uses the rule transformer to generate a rule 
candidate.

## Finishing Up

Once the halting probabilities reach 1.0, the learning cycle stops. At 
this point, the rule candidates generated in the previous cycles are 
combined using the ACT mechanism to produce the final rule. The model 
then returns this rule, which can later be applied via the mapping 
model to attempt to achieve the desired result.

## Integration of RLHF and Perplexity

The model uses a method inspired by RLHF, where a sequence of targets 
is generated and used for feedback in training. Perplexity is a key 
metric in this process, as the model seeks to generate experiments 
that yield surprising results, thereby maximizing information gain.

## Tensor Flow in the Model

### Scientist Transformer Stage

At this stage, the input tensors represent the current state of the 
sequence of experiments and their results. The scientist transformer 
outputs tensors that represent proposed experiments (action 
embeddings). These tensors are then used to generate and evaluate the 
next set of experiments.

### Rule Transformer Stage

The output of the scientist transformer (the selected experiment) is 
used as input to the rule transformer. The rule transformer processes 
these tensors and generates rule candidates, represented as sequences 
of embeddings. These candidates are combined using ACT to produce the 
final rule tensor.

### Mapping Model Stage

The final rule tensor, generated by the rule transformer and processed 
through ACT, is combined with the input collection tensor in the 
mapping model. The mapping model processes these tensors to produce 
the output collection tensor, representing the transformed state of 
the input.

## Loss Mechanisms

### Scientist Transformer Loss

The scientist transformer receives loss based on how close its 
generated sequence was to the target and how effectively the final 
rule moved the state toward the desired outcome.

### Rule Transformer Loss

The rule transformer's loss is primarily based on the accuracy of its 
output in producing the correct rule.

### Mapping Model Loss

The mapping model's loss is based on the effectiveness of the rule in 
transforming the input collection into the correct output collection.






































# Introduction

This document outlines a hybrid approach between reinforcement learning and 
supervised learning designed to help a model both experiment to gather information
and rapidly converge on an optimal outcome.

## Background

### RLHF

One very common existing strategy for fine tuning a transformer model is the 
Reinforcement Learning from Human Feedback strategy. In this method, a transformer
is given a prompt and generated multiple sequences - a human then selects the best
one. This is then incorporated into targets that can be used for teacher-forced
training. This, in effect, fine tunes an existing distribution into the most optimal
direction

### RLHF with Reward Models

One variant on the above is to replace the human with an AI-based reward model. 

Using this strategy, a human would be used to train a reward model that provides 
feedback on effectiveness of the distribution choices. The reward model then 
actually is used to train the transformer. This has also shown to be effective,
and shows the feasibility of automated feedback


The result, again, will be a target sequence we can use for training.


### Adaptive Computation Time

Adaptive computation time is a mechanism in which a computation that can take a 
variable amount of length is elegantly handled as a superposition of intermediate
results combined according to a halting probability.

### ARC-AGI challenge

The arc agi general reasoning challenge has proven to be extremely intractable to current
techniques. It involves generalized reasoning in a manner that is very unlike current 
training data. As such, it has proven extremely difficult to adapt current models and
ideas to handling the particulars of the task.

One of the major difficulties of this task is the need to respect an underlying, but not
specified, DSL. 

## Outline

**Model Jobs**

These are jobs that are the responsibility of the scientist model.

* Predict the best experiment proposal given the current experiment sequence
* Predict given the sequence and proposed experiment what the results will be.
* Predict given the current experiment sequence and the prior rules what
  we think the best rule will be.

**Logic Jobs**

* Given a cluster of experiments, run them all and get the results.
* Given a collection of results, find the perplexity of the results
* 

We propose a "Scientist Transformer" whose job is to run experiments and make
proposals that will maximize the amount of information that is gained when the
experiments are run.

The model operates in a 

This information is then used to in turn propose actions which will solve the problem.

### Model Jobs

### Information Flow

The raw input to the model will consist of a properly processed and concatenated
variation of the entire example set, plus its targetted output. This includes
input and output cases. The output case will be masked on the target, and like in
BERT the job of the model will include predicting the masked output.

### Jobs



There are two transformer models of note



### Assumptions

We are going to assume we have done some sort of suitable pretraining so that the model
understands the input format of information at at least a primitive level. 

We are also going to assume that the task we want to complete is to find a sequence
of actions $A_i$ that will when fed into a mapping function map the entirety of our
problem from it's input condition to it's output condition.

### Information flow 



### Terms

* experiment: An individual experiment consisting of a proposed Action.
* result: The result of an individual experiment.



* experiment_sequence: The sequence of conducted experiments and their results. 

### Model Design





transformer that accepts as input the input and output cases from the
example set, along with the proposed test case. The output case is then masked. 





## Proposal

We propose a novel mechanism designed to address parts of the ARC-AGI challenge.

In this mechanism, we train a transformer to produce and predict "actions": A sequence of
embeddings that has meaning in terms of transforming from an input state to an output state,
and receive "results": The results of executing the actions. This occurs over many rounds. 
Automatic sampling and selection occurs during each round, and the best path forward is selected
for further processing. 

What we propose will basically be a model that is designed to experiment. It tries to propose
experiments that result in a large gain of information. This is measured by considering the 
perplexity of the experiment result. Multiple experiments are proposed, evaluated, and only the
best result is kept. 

This then creates a sequence which can perform the feedback task that reinforcement
learning requires
 
### 


### Model

More formally, then. 
    



### Assumptions

We are assuming we are dealing with some sort of a transformer mechanism, or at least
a decoder. We are assuming we are attempting to generate a "rule", or sequence of
tensors, that will best map from the input condition to the output condition. 

Furthermore, it is assumed that the role of this transformer is to 

It is furthermore assumed that this 
This
mechanism, it is assumed, can be fed a prompt of some sort consisting of input
states and output states.

What we want to find is some sort of rule or action that brings us most effectively from
the input state to the output state for all items in the collection.

### Design

The transformer under consideration has the job of proposing experiments and rules to be 
tested against the input states. It rules, reviewing resultant
information, and making decisions that will then 

### The main generative cycle

The main g

### Experimentation Mode

### Prediction Mode


### Scientific method


## Design



We propose a mechanism designed to sampling 

### 






# Assumptions

* We have a model.
* It has been pretrained to produce sane output
* It has been fine tuned into the convergence task

## Distributions and Theoretical Justifications

This entire exercise is basically postulated on a single main point. That is the idea 
that:

* Once primary learning is complete, a probability distribution can continue to 
  be fine-tuned for a task by sampling and focusing the distribution towards
  the samples that did well.

Fortunately, this is the same assumption the policy gradient training
mechanism makes, and it has worked just fine for years. Hopefully
this will thus work good as well.

## Idea

Based on what we have tried before, we:

* Select a bunch of ideas that might get us closer to a solution
* Try all of them
* Keep the idea that gained us the most

Then do something more like this in the future. In essence, we 

* Sample from a distribution of ideas individual ideas
* Then test them and keep the best
* Then update our distribution to emphasize that case more

This is something that can be done with a transformer and a little
clever logic.

## Generative Learning Cycle

For the generative learning cycle, we predict a set of actions by continuing our sequence based
on what came before, evaluate the result of executing the actions, then keep the sequence that
produced the best results. This repeats many times, hopefully guiding us towards a solution.

At the end of this process, we can then do teacher-forced cross entropy loss using the model,
refining our distribution to better point us in the correct direction in the future.

The sequence we actually end up with consists of

inputs | action | result | action | result...

## Steps of the Learning Cycle

The learning cycle consists of three distinct steps. These are:

* propose_actions: Propose actions that will get us closer to a solution
* execute_actions: Execute them. We get a result
* select_continuation: Select the best continuation, and continue generating from there

**propose_actions**

In this step, we essencially sample a bunch of embeddings, a bunch of times. We end up 
with a set of actions we can execute against the input. We will need to execute them,
then process the output

**execute_actions**

We execute the actions. We get a set of results back, one per action. 

**select_continuation**

During this step, we evaluate the results, and select one result set that we will
be using to continue the sequence. Once we select the continuation, we can continue
generating from that point forward.

The process of actually making the selection is a bit nuanced, and will be discussed more
in a second

## Nuances of selecting the continuation

Basically, this step wants to select the "best" proposed actions and continue the 
sequence from it. This will both ensure the loss focuses the distribution in a more
optimal direction, and will ensure we have the best chance moving forward of predicting
additional good features. 

But what, exactly, does it mean to select the "best" proposed action? The answer may
not always be the one that results in the best proposed rule - indeed, sometimes we might
get better results by pursuing gathering information and giving our actual rule later.

To handle this, we include annealing in our selection process, and change between selecting
the continuation which was most surprising and the one that has the best rule metrics.

### Annealing

An annealing term, which can vary between 0 and 1, is used to control the annealing
probability. This is the probability that we will focus on exploration, or on getting
the best action possible

The annealing term will be scheduled based on the number of completed cycles, and 
is expected to eventually end up 

### Exploration selection and perplexity

When exploring the problem, we need to somehow figure out what action produced
results that were surprising. This is accomplished by means of computing the perplexity.

We take each result, compute the normalized perplexity the model would associated with the result, and
select the continuation with the highest perplexity.

The idea here is that a high perplexity meant we were surprised to see it, and hence it provides
the most information.

### Best action selection

The other option is to select based on what looks like the best possible action.
This can be evaluated a number of ways - maybe we know what the action sequence
should look like and perform cross entropy.

Or maybe we known what the outcome should look like, and we track how well the 
defined action reconstructed it. 
