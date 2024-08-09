# What is the transform executor?

The transform executor is the portion of the model responsible for actually executing a transform. It will
need to deal elegantly with ideas such as decoding the transform directive into pieces,
and actually executing the transform sequence.

It should be noted that the objective of the entire process is to get the right transform
directive to pair with the transform execute to solve the problem.

## Transform Executor Contract:

### Input Contract for the Transform Executor

The transform engine will be fed by the following parameters. 

**start_condition**

This is the starting condition of the latent space. Given the format of the problem,
the main restriction on the latent space is a need to support sequences. It will likely
consist of

* latent_space: Shape (batch x collection x 1 x  ...)
* latent_mask: Shape (batch x collection x 1)

Where the 1 is the first input into the sequence. 

**targets**

This is the target latent space we are looking to reproduce. It again consists of a 
masked collection of sequences. 

* latent_space: Shape (batch x collection x sequence x ...)
* mask: (batch x collection x sequence). 1 when we should include.

**transform_directive**

The transform directive is the last parameter we need. It has pieces

* instructions: Shape (batch x instructions x embedding)
* return_probabilities: Shape (batch x instructions)

### Output contract for the Transform Executor

The transform executor pledges to return

* latent_result: The results from executing the transform directive
* mask: The mask, with 1 meaning active. 

### Logic contract for the Transform Executor

The Transform Executor pledges that it will:

* create an auxiliary set of results, then map them into the output slots.
* utilize teacher forcing during training with sequence targets or self-excitation during 
  evalutation with sequence outputs.
* provide an output consisting of the decoded latent space and the mask to go along with it.

## Implementing a Transform Executor

### Decoding a transform_directive

Decoding a particular transform directive operates sort of like decoding a sequence of sentences
using an NLP transformer, and letting it see the previous sentences by cross-attention, but also summarizing
each sentence into a single vector. We use a specialized multidimensional transformer for this task.

The first thing that needs clarifying is what exactly is fed into this specialized transformer. Consider
a "starting_state". This state will consist of the previously generated output - an additional complication
is that during training this state will be teacher-forced, while in evaluation it is self-excited. 

From here, we concatenate the instruction for the step onto the state, then feed this into the multidimensional 
transformer,  along with a context tensor consisting of the auxiliary responses from previous generation cycles.
We get out the next auxiliary response, which will have dimensions reduced back down to standard. To this
we concatenate the NEXT instruction, and feed it back in again. We end up being able to generate a sequence
of embeddings - the new auxiliary responses - for this cycle. 

A given cycle ends once we have exhausted all response probability for the output slot. Once this is done,
we perform a weighted sum of the relevant probability and add it to the output slot. We also concatenate
the auxiliary responses for the cycle onto the context auxiliary responses. 

This repeats until all instructions are decoded. At this time, a mask is created based on the last
completely decoded output slot, with 1 indicating generated and 0 not. The outputs, and the mask,
are then returned.

### The specific implementation

Everything up to this point has been generic to tensors of shape 
(batch x collection x sequence x ... x embedding). Now we need to get into the specifics
of what kind of tensors we are working with. The latent space we work with
will be represented as:

* latent_space: (batch x collection x sequence x  num_embeddings x embedding_dim)
* latent_mask: (batch x collection x sequence)

This means for every batch, and collection in each batch, we will need to figure out
how to integrate decoding the transform terms into a task of generating sequences of 
latent spaces.

### Multidimensional Transformers

A specialized transformer will be used for the task. The transformer will be 
capable of either performing attention looking along the sequence dimension,
in a masked generative self_attention manner, or along the num_dimension dimension, in
which case it operates in a encoding mode with no masks. 

This is followed up by a feedforward layer like normal. It will allows processing of
an entire latent space as a task consisting of generating successive latent space outputs
from the input. These steps act as the "self_attention" behavior of a decoder.

In addition to all this, however, there is also a context cross-attention parameter.
The purpose of this will be described shortly.

We incorporate mask logic where appropriate. 

*TODO: Flesh out mask logic better.

We will call this the Latent Decoder Transformer

### Outcome

At the end of the whole shebang, we get the outputs and the auxilary responses as returns.