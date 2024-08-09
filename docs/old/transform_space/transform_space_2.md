# Introduction

The transform space is, conceptually, the space in which all possible transform 
directives live. A instruction can be decoded into a transform space operation, 
or we can view the results of applying a transform space.


## Overview

We view transform creation to be about 


## Important tensor

**Cognition Engine Interface**

These tensors are used for communication with the higher-level cognative engine.

* cognition_command: Shape (batch x embedding_dim). Represents a demand to create a transform
* cognition_query: Shape (batch x num_embeddings x embedding_dim). Represents question about transform context and history

**Transforms**

These represent an actual transform that can be executed.

* transform_directive: Shape (batch x instruction_order x embedding_dim). A interpretable directive
* transform_directive_mask: Shape (batch x instruction_order). A mask 

This is feedback we get when executing a transform directive. We use the cognition command as the
query

* transform_feedback: Shape (batch x instruction_order x embedding_dim). Feedback from executing the directive
* feedback_summary: Shape (batch x embedding_dim)

**Transform history**

This represents a sequence that is generated, and contains many of the previous pieces
plus a few new bits

Shape: (batch x num_generated x instruction)

* transform_history_sequence:
  * Concatenation of transform_directive, and transform feedback. 
  * Shape: (batch x num_generated x instruction_order x embedding_dim)
* transform_history_mask: Masks inactive instruction elements
  * Shape: (batch x num_generated x instruction_order )

## Transform generation by generating a sequence

Transform generation basically is treated as a specialized generative transformer case. We look to 
tell based on the previously generated sequence, and current command, what will be the next element we 
need to generate.

A complication exists, though, in the form of the instruction order and nested generation. Each proposed
transform can be a variable number of dimenisons, and as such a second generation task is needed to produce that sequence.
And that generation task will need to be handled as well

## TransformDecoder

The transform decoder is the primary item that will make a new transform given the history
and context.

**Dependencies**

* summary_attention: Makes a summary of the memory based on the cognition command.
* halting_logits: Makes halting logits out of the transform during each step.
* subdecoder: The transformer-based subdecoder, which decodes each instruction as available.

**Accepts**

* cognition_command: (batch x embedding_dim)
* transform_history_sequence: 
  * The previously generated transforms and their feedback. 
  * Shape (batch x num_generated x instruction_order x 2*embedding_dim)
*transform_history_mask:
  * The mask for the transform sequence
  * Shape (batch x num_generated x instruction_order)

**Returns**

* transform_directive: The directive. Shape (batch x instruction_order x embedding_dim)
* transform_mask: The mask. Shape (batch x instruction_order )
* halting_distribution: The halting distribution. 
  * Used for training sometimes. 
  * Shape: (batch x instruction_order)

**Design**

CONTEXTUALIZE_MEMORY:

Before proceeding forward, we concatenate each memory unit with an additional memory
feature generated based on using the summary_attention mechanism. We do this along only the 
instructions dimension. This produces a summary of sorts based on the entire instruction dimension.

This is then concatenated onto the transform_history_sequence, producing something
of shape:

Shape (batch x num_generated x instruction_order x 3*embedding_dim)

Each individual unit now has some information on the overall behavior and how it is
important.

DECODER:

We then proceed to apply the transformer decoder process in sequence based on slices
removed from the memory. If we are generating instruction 0, we remove the instruction 0 set from 
memory, and perform attention. The output will be the new instruction

That instruction is then fed back into the decoder with the updated context slice, completing
the loop. 

HALTING:

During this time, we will be generating halting probabilities.

Halting probabilities have a noticable effect on the mask. In specific, the sum of all generated
halting probabilities is subtracted from 1 to make the mask entry. This means generating halting
probabilities early will shut down any ability to generate more terms. 

Once all halting probability is exhausted, generation stops for that batch. Once it is exhausted
for everything, everything also stops.

The mask can then be compared to a target mask during training or pretraining in order to train
this behavior. 

HISTORY:

The history tensor is not updated by this portion of the model, but by a higher level process

## TransformReader

This reads the transform history and converts it into something like a cognition thought.

**Dependencies**

*

