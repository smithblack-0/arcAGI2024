# Need

* To decode multiple modes of data
* To do so in a manner that a transformer can process
* This includes:
  * proper positional information
  * Sequence information
  * Mode information

# Design: Decoding Modes

In broad strokes

* During decoding, we have multiple modes of decoding we can operate in. These modes are probabilistic
* Modes have a CONTEXT consisting of. This is defined right after the mode switch.
* Every N tokens we have the option to switch mode
* During a mode switch we can choose to STAY, CHANGE, or END with an associated probability.
  * STAY keeps generating with the current context set
  * CHANGE creates a new context set, with the given probabilitie
  * END indicates the end of the generative process. Like End Of Stream.
* During a STAY branch:
  * Continug generating context as before. But maybe trim low probability generators
* During a CHANGE br

## Modes. 

Multiple modes of content generation will be supported by the model, and will be produced in
sequence blocks of similar mode content. Conceptually, the model may consider itself to be 
responsible for generating blocks of continous content that are in the same mode and will
be evaluated in the same way - for instance, responsible for generating a string of 
text followed by a picture.

The mode the model is operating in is a probabilistic superposition of all possible modes,
with some trimming of unlikely probabilities. The mode can be switched every N tokens - this
region is known as a Mode Block. The state of being able to change mode configuration is
a Mode Change Chance. A particular configuration of mode start point and mode context is known
as the Context Configuration.

Every N tokens the model is expected to produce a ModeControl embedding. This embedding both
is expected to contain context used for building the Context tensor, discussed later, and the 
information needed to select whether to STAY with the current configuration or CHANGE to a new
mode configuration with new context. It also is the only time an END token can be generated

## The Mode State Collection

It is very important for the generation of context information that the mode we are operating
in can be represented probabilistically, and that we can tell how many tokens ago that mode
was started. We also need to know what context to encode. This is the point of the mode 
state collection. Lets get more familiar with the tensors involved.

With it being the case that "mode_items" indexes the generated Mode embeddings, not the output
embedidngs the mode produces, and "modes" indexes the modes available, the tensors of concern
are the mode_probabilities, the mode_start_position, and the mode_embeddings. These respectively
control something about what mode we are generating in, when we started this generation process,
and what context to use. The mode probabilities are later used to superimpose all possible 
context tensors - with reasonable trimming of unlikely probabilities.

* mode_probabilities: 
  * Shape: (... x mode_items x modes)
  * Purpose: Indicates how active each mode context is using a probability
  * Details: The sum of the items + modes dimensions is equal to one. 

* mode_start_position:
  * Shape: (... x mode_items)
  * Purpose: Indicates where the main sequence stood in terms of number of generated tokens
  * Details: Is an integer tensor

* mode_control_embeddings:
  * Shape (... x mode_items x embedding)
  * Purpose: Is what the mode control flow is run from. Also is expected to contain information
             that can be used to build a specific context

* sequence_length:
  * Shape: (... )
  * Purpose: Tells us how many contigous mode sections (NOT CHUNKS) we have made

# Mode Contracts

Each mode that will be defined must present to the decoder several features. These
are part of the decoding contract they must satisfy to live within the same mode.

All contract pieces must be satisfied for a mode to be supported

## Updating the Mode State

The mode state collection can be updated only every N tokens. Every N embedding generated by the
decoder mechanism is expected to be not an ordinary embedding, but a specialized mode_control_embedding
that will be interpreted by the model.

When a mode_control_embedding is produced, it is the case that it is used to project and evaluate
a set of probability tensors. Lets learn about them. With L being the number of supported modes, the
two tensors we make are the flow probabilities, and the mode_select_probabilities. Lets learn about them:

* flow_probabilities:
  * Shape: (... x 2)
  * Purpose: Controls how strongly we keep the current mode state or create a new mode state
  * Details: Sum along dimension 2 is 1.0
  * Details: We have a KEEP probability and a CHANGE probability

* mode_select_probabilities
  * Shape: (... x L)
  * Purpose: Controls how strongly we generate context in each mode when starting a new mode state
  * Details: Adds up to 1.0 when summed along L
  * Details: Each of the L probabilities is associated with one generation mode.

To update the mode_probabilities, we produce the two probabilities listed above. Then, we 
perform the following sequence of actions, involving shutting down the existing mode selections
and integrating the new choices.

mode_probabilities *= STAY
mode_probabilities = concat([mode_probabilities, mode_select_probabilities*CHANGE])

The remaining two details are to update the mode_start_position, mode_control_embeddings, and 
sequence_length tensor. With it understood generation index represents the index associated
with primary generation, these are updated respectively as follows:

mode_control_embeddings = concat([mode_control_embeddings, mode_control_embedding])
mode_start_position = concat([mode_start_position, generation_index])
sequence_length = sequence_length + CHANGE
## Mode Shape Contract

### Mode Shape Creator Contract

The Mode Shape contract is required to be fufilled by all supported modes. It consists
of defining some information needed to generate positional encodings.

**Purpose**

In order to decode with the correct context, we will often find that we need to know
the shape of the final output. For instance, for an image, we might need to know
the image is 200 x 200.

We also need to be able to train that. As such, we create a layer whose is responsibility
is predicting the shape of the mode output, if it is needed, and which will provide a
distribution we can train against.

**Accepts**

* control_embedding:
  * Shape: (... x embedding)

**Returns**

* shape_contract: 
  * Shape: (... x L), where L is the number of supported shape dimensions.
  * A value of zero on any dimension means it is ignored. 
  * A value of zero on all dimensions means do not enforce shape encoding at all.
  * Will be a tensor of integers.
* distribution: 
  * Any
  * Contains information which can be used to evaluate how the choosing of the shape
    went.

**Design**

How this functions is that internal models can be used to create a distribution representing
the various possible shapes as probabilities, which can then be selected from. This distribution
can then be returned for training. 

It is likely easiest to have the shape_contract consist of the most probable entry in the
distribution.

It is expected that shape information will be available in the training data and can be made to
interact with the distribution to train the model.

### Mode Shape Embedding Contract

**Purpose**

We need to be able to convert a shape_contract into an embedding for integration
during training

**Accept**

* shape_contract: 
  * Shape: (... x L), where L is the number of supported shape dimensions.

**Return**

*pos_embedding:
  * Shape: (... x Embedding)

## Mode Encode/Decode Contract

For the mode Encode/Decode contract, we need to provide a means to create embeddings
when operating in a particular mode, and a means to produce 

## Positional Encodings

Positional encodings will be decoded in one of a few manners, depending on the 
number of active dimensions.

The positional encoding mechanism will be quite flexable, and capable of addressing all
L available dimensions. To get positional encodings, a "shape_contract" must be provided 
along with the indices to generate at.

What will be returned will be, essencially, the indices of a flattened version of the ND encodings
for the shape contract. This means, for instance, that if you had a 10x10 image, and you asked
for the 12th encoding, you would have gotten the encoding for row 2, column 2. Shape specifications
that are zero do not generate any information. The entire collection is concatenated into one
embedding with assignments to dimensions on various portions of it - for instance, with 4 supported
dimensions 1/4 of the embeddings is dedicated to each dimension. Once all shape portions
are generated, we stop providing positional embeddings.

In the special case of being given a shape_contract filled with lengths of zero, we will keep
generating positional encodings forever.

**Accepts**

* shape_contract: 
  * The shape contract to generate on
  * shape: (... x L)
* starting_position:
  * The starting position for the mode
  * Shape: (... x items)
* positions:
  * The positions to generate encodings at
  * Int positions from start.
  * shape: (... x items)

**Returns**

* positional_encodings:
  * The positional encodings for the sequence
  * Shape: (... x items x embeddings)

## Generating Modal Context 

The point of all this work is to generate context that is specific to the mode we are working in,
such as information about the positional coordinates of an image being generated. This allows 
text and nd data to coexist within the same model.

To make this happen, once we hit a mode_control_embedding and process the mode state update, 
we select the K most probable modes to inject as context over the next mode block. We use that
to make the various shape contracts for these K case, and we then over the next block of generated
tokens query all K of the different mode encodings with their required information, and superimpose
the results together using their probabilities. This can then be injected as context once the token finishes
generating in the main decoder model.

## Stopping

One additional detail that has not been brought up yet is how to stop generation. We go ahead and 
predict the probability of stopping using the mode_control_embedding anytime a new one is generated,
and include in the training data practice on when to generate this token

# Training 

Lets now talk about training the whole mess. This design can, basically, be trained by 
a fairly standard suite of next embedding prediction with teacher forcing.

Lets assume we are working with some sort of input encoding, dealt with seperately but perhaps
encoded similarly, and an output target which consists of T pieces of multimodal information that can
be flattened into embeddings. We wish to create training targets that can used to train the model
using this information. How do we do it?

## Data



We suppose that we wish to encode as targets a sequence of modal information. It might consist
of text then images then text, or text then text, or whatever.

