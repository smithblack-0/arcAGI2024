# The scientist model

The scientist machine learning model is designed to operate effectively when a scenario exists wherein we will know what a right answer looks like, but not necessarily how to reach that right answer.

Like a scientist, it will propose and test hypotheses in order to zero in on what the right answer is. As such, it must have access to input and output data in order to test whether it's current hypothesis completes the task.
It also must be operating on a task for which completion of the task can even be judged. It is best applied for pattern matching,
wherein a input collection of examples is examined and a pattern is extracted.

The model, broadly speaking, handles two distinct tasks

* Proposing candidate pieces to build into a solution
* Choosing from the existing pieces a better solution

This is iteratively repeated as we converge towards a solution

Major pieces of the architecture include

* Central Cognition Unit
* Environment Actions Unit
  * Memories Association Unit
  * Hypothesis Testing Callback
  * Feedback Formatting Unit
* ACT accumulate and output

## Central Cognition Unit

This is one of, if not the, most important parts of the model. The Thought Generator looks back
based on history, looks at the feedback that has been received, and comes up with a new state to
work with.

The point of this is to pass the previous generated state and the feedback that was generated in an 
format-agonistic way into the model. With any luck, this will allow one thought generator to be pretrained
and reused.

**Dependencies**

* decoder: A transformer decoder. 
  * Should support encodings float masking.

**Premise**

* A stream of thoughts and a stream of token embeddings are not as different as first appears.
* The model should be able to decide what output goes with what channel

**Accept**

* Memories: The memory tensor. Almost always a collection of states. Shape (batch x N x embeddings)
* Feedback: A Feedback tensor collection.
  * Feedback tensors can come from prompt or test results
  * Prompt feedback is marked
  * Hypothesis feedback retains information on the originating hypothesis.
  * Shape (batch x L x embeddings)
* Feedback Emphasis: A mask operating on various feedback elements.
  * Provides an additional emphasis channel.
  * Shape (batch x L)

**Return**

* State: The updated state tensor. Shape (batch x embeddings)
* Memories: The updated memories tensor. Shape (batch x N + 1 x embeddings)

**Design**

A torch transformer decoder has everything we need. We just have to hook it up

## Environment Actions Unit

Interacts with the environment to run tests and produce feedback



### Hypothesis Expansion Unit

The hypothesis expansion unit forms a number of hypotheses and associated tests in order to better
understand the problem and move towards a solution

**Dependencies**

* test_expander: A 

### Test Expansion Unit

The test expansion unit creates a variety of encoded "tests" which we wish to run against
the environment. 

**Premise**

* Multiple tests are a good idea, as they make gradient descent much easier to train
* Only the top tests need be run. 

**Accepts**

* State: The state tensor. Shape (batch x embeddings)

**Returns:

* test_encodings: 


* create a hypothesis 
* test that same hypothes



# Hypothesis Testing Unit pieces

The hypothesis testing unit acts to, collectively, disconnect the specific task being performed
from the act of thinking about how to solve a problem. For each environmental channel, it assigns
all memories a probability using a sigmoid logit activation - these probabilities can then be thought of as the probability of that 
memory being relevant to that channel. 

This then results in a sequence of memories and a set of probabilities that can be taken apart and dispatched
to each environmental channel. The result can be interpreted as a set of embeddings with a mask, a series of actions,
or something else.

## EnvironmentActionEncoder




It takes an encoded hypotheses and memories
then assigns each memory a probability a


## Central Cognition Unit

The central cognition unit is basically a transformer acting in generative mode. It controls and proposes solutions to the problems, 
including intermediate steps, just thinking, and possible solution pieces. We treat cognition as a process of generating a "stream of thoughts"
that can sometimes be decoded into actual real-world solution representations, and sometimes might just represent intermediate thoughts

**Premise**

* We can process thought in a very general latent representation engine as a transformer decoder
* Each "thought" can be, under the right circumstance, addressed to the outside word
* These thoughts can be decoded in such circumstances.

**Accept**

* state: The current state. May be in starting position or also contain feedback. Shape (batch x embeddings)
* memories: The memory encodings. Expected to be previous outputs. Shape (batch x N x embeddings)
* feedback: Feedback tensor. Provides information on the environment, interaction, and results. 
  * Shape (batch x L x feedback)

**Returns**

* state: A embedding representing the next state. Shape (batch x embeddings)
* memories: The new memory encodings. Unless there is reason not to be, usually the concatenated state
  * Shape (batch x N + 1 x embeddings)

**Design**

This is in most ways the simplest unit to design. Just use a transformer decoder and you are pretty much done. 

## Hypothesis Expansion Unit

We need a mechanism to expand a state into a set of hypothesis to be tested. 

**Dependencies**

* embedding_dim [int]: The size of the embedding dimensions
* num_hypotheses [int]: The number of hypotheses to generate

**Accepts**

* state [torch.Tensor]: A state tensor containing current information. shape (batch x embedding_dim)

**Returns**

* hypotheses_action_encoding [torch.Tensor]: A tensor containing the encoded hypothesis actions. 

**Implementation**

A LSTM with a reshape or a feedforward are both perfectly adequete options

## Hypothesis Testing Unit


## Hypothesis expansion unit

Once a state has been newly created, it will hopefully contain new insights into how to solve
the problem based on the feedback. 

Based on this idea, we create a set of *hypothesis* - variations built on the current state - which
we then encode

We use that to create *hypothesis* - varations on the state that might lead
towards a solution - then try them out to produce feedback. 


* Expand as hypothesis
* Evaluate hypothesis
* Encode evaluation as feedback