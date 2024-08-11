# Introduction

This contains the documentation and specifications for getting the cognition model ready to go. 
It discusses what is going on in each subunit, what is needed for operation, and why. 

## What does ARC-AGI require?

ARC-AGI requires some significant lateral thinking to beat.

**Pretraining and data**

One extremely major issue is the amount of training data is not up to the task at hand.

When training a model novel on a task, it is common to use billions of examples in the
training process. This is not available. Training directly will thus almost certaintly result
in overfitting and uselessness.

**Lack of significant pretraining data**

One way to get around this is to pretrain a foundation model then use transfer learning on the
task itself. Unfortunately, this is not such a great solution here. There are really no significant
bodies of data that might be applicable that could be used to pretrain a foundation model.

**Novel requirements**

The need to correlate information across collections also adds quite a bit to the difficulty.

## How to challenge ARC-AGI

We challenge the task by a combination of a specially designed multimodal transformer model,
a specialized pretraining pipeline, and transfer learning for the entire task.

## Mixed Schema Encoding

All data, which can and will include a mixture of flattened int grids and text,
will be encoded in a common embedding-based representation referred to as Multimodal
Schema Encoding

### What is Mixed Schema Encoding?

Mixed Schema encoding aims to let multiple modes of operation coexist, and generation smoothly
choose between them, using probabilistic techniques, a stateful mode tensor, and a short probabilistic
history of time since mode start.

The main objective in Mixed Schema Encoding is to manage to encode Mode context, like metadata,
and important information, like positional encoding, in a manner that is common when decoding
or encoding.


### Mode Vocabulary and Common Metadata Vocabulary 

There are two very important vocabulary features that must be specified,
and important restrictions they bring about. This is the Mode Vocabulary and 
Common Metadata Vocabulary. 

**Mode vocabulary**

The mode vocabulary is, quite simply, a vocabulary that is associated with the various
generation modes. 

* raw data: Operating in raw data format, it is one of N mode tokens, where N is the number of
  modes, plus a content token.
* Embeddings: As embeddings, these consist of a similar selection of embeddings

**Common Shape Vocabulary**

The shape of an input, or an output zone, is fixed. This information is specified
using the Common Metadata Vocabulary. The Metadata Vocabulary is of fixed size, and that
size determines the maximum number of discrete entries that can even be encoded in a
particular mode. 

Conceptually, we assign one Shape case to each token in the vocabulary. Lets say we
have a vocabulary that consists of 1,000 Shape tokens. We could use that to represent:

* A sequence of text that has max length 1000, where each shape token is corrolated directly
  with a length
* A sequence of 2d data with shape 10x10, where the flattened shape are corrolated with a length
* 



### Walking through encoding a mode.

On the raw data side of things, encoding a mode requires a data entry formatted in a particular
manner. In detail, we need a dictionary listing

* type: The type of the mode
* shape: The shape of the mode, a list of ints. Variable length depending on the data.
* payload: The unencoded payload.

There also exists another integer feature, known as the sequence position,
that will be externally provided and indicates the position of the mode being decoded.

**Encoding the payload**

Step 1 is to encode the payload. Using a mode-specific encoding mechanism, we encode
the payload. We end up with the ModePayloadEmbeddings

**ShapeToken**



**Encoding the metadata**

The metadata is then mixed into the

** Encoding the Payload**



**Encode shape**

We promise that when encoded, the mode will consist of a tensor of 

  * [ModeDeclarationEmbeddings], SequenceEmbedding, ShapeEmbeddings..., PayloadEmbeddings...

Where it is the case the ModeDeclarationEmbeddings are associated with the type, the
SequenceEmbedding is an external factor controlled by the sequence position, and the 
ShapeEmbeddings are associated, one each, with the shape metadata channel.

**Context tensor**

Part of the encoding process will be generating a Context Tensor for the mode. This consists
of generating a tensor that contains Nd Positional Encodings, Sequence information, and Mode
information, then adding it to the entire payload embeddings sequence.

The context tensor will have the same Sequence and Mode term for all parts of the sequence.
However, the positional encodings portions are different. They will consist of, basically, generating
positional encodings for the Shape, and flattening them.

This context is added to the EncodedPayload 

### Mode Support

To support a particular mode, you need to provide a vocabulary size, a shape size,
a positional encoding mechanism compatible with the shape, and optionally a training padding
specification. 






[]

The way this is done is to allow probabilistic creation 



### Vocabulary Details

The first thing that bears talking about is the supported vocabulary. There is a common vocabulary
models need to share and understand, then discrete vocabularies per mode, with the idea that
a logit is returned whenever a vocabulary is computed. 

**Common Vocabulary**

The common vocabulary is used for indicating mode informat

**Mode Vocabulary**

There exists a mode vocabulary. For each supported mode

**How decoding works**



### Multimodal Schema Encoding




It consists of:

* A metadata vocabulary. This can be used to encode
  * A null or no change token.
  * mode_start, mode_end, tokens for each supported mode.
  * Numeric information up to a certain point.

* An Embedding Structure that the model views and interacts with:

* A data input format:
  * List of Modal Entries.
  * Modal entries contain:
    * type: The type of the mode
    * metadata: A list of integer metadata. May be empty
    * payload: An unencoded payload.


**How does encoding work?**

Start with list of unencoded structured data in type, meta, payload format.

* type: The type of the payload
* meta: The metadata for the payload. A list of items to convert to meta tokens.
* payload: The actual payload itself. A list of items to convert to 

Based on the type, encode the meta


The Schema Specification is designed to allow multiple modes of information to coexist
within a single model as a concatenation of emebeddings, and to notheless still elegantly
support generation of such embeddings in a transformer-type model. It does this
by specifying the type of information being worked on, details on important context, and
injecting additional information within a given mode. The pieces can be concatenated together

By correctly using "Schemas" with certain rules on probability during a decoding
process we are able to effectively encode and decode grid based positional information.

The physical encoding as going into the model consists of chains of:


**Mode requirements**

In order to support a mode

* MetaDataConversion:
  * Must be able to convert provided metadata of the mode into a set of context length embeddings
  * Must be able to convert the embeddings back into metadata by a probabilistic process.
* PositionSupportEncoding
* MetadataSupportEmbeddings: Must be able to produce 
* 

Each mode to be encoded needs to be able to provide several things. These are an:

* ident embedding. This will be added to every payload embedding
* 

**What are the technical details?**

Lets walk through the process of encoding something in schema specification for
processing by an encoder of some sort. Consider a data that consists of a sequence
of multimodel content, such as text interwoven with pictures or intgrids. 

The journey starts as a list of tuples. Each tuple contains the 

Different modes of a model are placed within specialized schema zones indicated
\by different tokens. For each mode we want to integrate into the model stream



that each unit of mode payload should consist of a mode start
token, a sequence of prelude embeddings, a sequence of payload embeddings, and a mode end
token:


[BeginTextSchema] "[Id: 1] [EndTextSchema]
[BeginGridSchema] [Id: 2] [10] [20] [Tokens] [EndGridSchema]


The number of Prelude Embeddings required is specific to the mode being encoded,
and typically would be utilized to encode information such a sequence length or grid size.
Following them are the payload section itself, which should contain positional encodings. 

**What are the Prelude Embeddings?**

The prelude embeddings must include an identifier, the ID embedding, as its very first
entry. It may then contain a number of additional embeddings, or none at all. These can 
be used to encode additional metainformation such as grid size or sequence length. The number
of expected additional prelude embeddings is an important factor associated with each Mode

**What are the Payload Embeddings?**




**What are the Payload Embeddings?**

**Encoded**

Data is encoded in terms of an extended vocabulary, consisting of the vocabulary
from all the different modes of operation. There is also a mixing of special tokens
known as "Schema" tokens that indicate what kind of information is being generated.

Any block of encoded information is opened by its SchemaOpen token, and closed by a related
SchemaClosed token. Additionally, immediately after the SchemaOpen token there may be several
tokens reserved for details about the schema before the main payload. This can be used to encode,
for example, dimensions of a grid. This
The boundaries of a Schema Zone may have special additional rules associated with it. 

**What is it**



Basically, all modes of operation are generated at the same time, but they are merged
together only according to their Schema Probabilities. Schema Probability must be
reserved before generation can occur: This means producing a certain SchemaStart token for the 
associated schema and locks that much weight onto the schema. Predicting a SchemaEnd token,
meanwhile, will release that probability. The total probability must not exceed one, and so
if there is insufficient probability we will clamp any attempt to open a schema with insufficient
probability to the highest value that is available.

Opening and closing the 

All modes share an embedding space, but the final logit and softmax prediction is different between
each mode. Additionally, a specialized additional layer predicts "Schema" tokens such as 
StartText, EndText, StartIntGrid, EndIntGrid, and maybe other modes if the model is extended in the
future. There is also a NoChange token. 

Each mode has a reservation slot associated with it, in a probability tensor. When generation
is desired in 

When a mode sta

The start of a mode is indicated by the production of a specialized Schema token which
indicates the mode we are operating in. 

* SchemaStart and SchemaEnd with probabilities
* When one schema is going, it locks out another schema from working.
* Some Schemas (gridint) can place constraints down. These will provide extra 

**MultimodalEncoding**

All data, text or grid based, is encoded within the same sequence of embeddings. Special
tokens, called Schema tokens, are expected to be used to declare the start or end of 
a particular data type.


Additionally, when producing under a given schema it is expected 
that 



Conceptually, you can imagine declaring a 


**Common Modal Encoding**

An i

All representational data shall be encoded in terms of a batch of collections of sequences of embeddings and an
associated mask tensor. This is where the "collections" dimension is associated with the idea of input-output
map pairs, and the sequence dimension is assiated with discrete inputs or outputs. However, sequence is not
guaranteed to be of length two during pretraining tasks.

A single encoding of such data has shape:

* data: (batch x collection x sequence x embedding)
* mask: (batch x collection x sequence)

**Examples vs Holdout splits**

Training data from a raw task can be thought to be divided into
an 


* (batch x collections x sequences x embeddings)

An input to the 

** Inputs vs Target Splits**



** Examples vs Holdout Split**


All data shall be processed into batches 

### Model Contract



## How we challenge the ARC-AGI 

We propose a model that operates by defining a "rules" tensor which is common among all
input cases and directs the model how to decode the case. Alongside this, we ensure that
pretraining tasks involve a requirement to corrolate data across multiple 

## Latent Conversion

### Text converter

TODO

### Grid Converter

TODO

### LatentSpaceConverter

TODO

## Training Generators

### Data

There exists

* Example dataset
* Holdout dataset

We learn from the 

* Example dataset

Then try to apply what we learned to the 

* Holdout Dataset

### Tasks

* Masked Discover Transform Task
  * Generate a random sequence of transforms on a grid, and a group of random starting grids
  * Apply transforms. We get a collection of grid sequence. These are the targets.
  * Mask out some number of elements from each collection. These are the examples. 
  * The masked examples are the targets.
  * Analogous to Masked Word Prediction.
  * Loss based on success in the holdout set.

* Dropped Discover Transform Task
  * Generate a random sequence of transforms on a grid, and a group of random starting grids
  * Apply transforms. We get a collection of grid sequence. The last element in the sequence is the
    targets.
  * Drop some evidence between the start and end of each sequence.
  * The model loss is 
    * based only on how effectively we were able to find the output piece of evidence
    * It does not matter how many intermediate steps we took.

* Density Sorting Task:
  * Given 3 colors in a %50 filled random grid. with a "color priority"
  * Move all nonempty to the bottom of the grid, and sort the colors from
  * lowest priority to hightest priority going up. 
  * Think liquids settling out due to density.
  * Example task: You are given some examples and need to learn the rule
  * Eval task: You need to apply the rule you have learned to a new situation.




* Separate Puzzle Task:
  * You are given a set of collections consisting of puzzle pieces. The pieces have
    been shuffled between the collections. There are extra pieces.
  * Your example task is to solve for each puzzle, then provide the unused pieces.
  * Your eval task is to take the knowledge that has been encoded, and use it to 
    figure out the last puzzle. This puzzle will have been made out of some of the extra
    pieces we identified. 

* 

* 
  * 
    
* Tile Pattern Task
  * Given a tile starting condition, and information on how many dimensions and 
    what to change as you change dimensions
  * Create a matching tile set.

## Similarity

We train a certain similarity task. This task is:

* Tell whether or not a given sequence is part of the same collection

**methods**

* Cross validation with ensembles is used to get a collection we can average.

**usage**

This is used while tutoring to ensure that the entire process does not go off the
rails.

## Common

The Ruleset Evaluator is common between the models and the tutor.

## Tutoring

Basically, we attempt to keep our similarity in limits while generating synthetic examples.
We also attempt to ensure the model can only solve about 50% of the problems, so it will have
good gradients.

### Generation

We create, from a noise distribution, input example cases and a ruleset. We then
feed the input cases through the Ruleset Evaluator. This gets the output cases.

### Similarity

We compute the similarity between the generated

* 

* Similarity evaluation - between collections. Trained triplet
* Collection 

## Model Concept

You have three primary sections to the model, which are
updated 

## Main Model

* Decoder mechanism makes the most sense
* Examples, Bottleneck, Predictions 

### Memory Management Unit

* Handles retaining memory of previous transform attempts
* Handles remembering how the results behaved. Including loss
* Handles answering memory queries, including collection mask
* Collection dropout mask
* Responses should be batch normal distributed.

### Environment Querying

* Has access to the example ruleset.
* Handles querying from the example ruleset.
* Collection dropout mask
* Responses should be batch normal distributed.

### Main Cognition Unit

* Has access to QueryingUnits
* Generates 
  * Context Tensor
  * Query Tensor
* Attempts to reduce uncertainty as fast as possible by hypothesis-check process
* Context Tensor acts as a sort of bottleneck
* ACT-like output

### Rule Distillation Unit

* Uses Context Tensor, 
* Tries to generate optimal ruleset for entire problem.
* Maintains information on ruleset length.

### Rule Execution Unit

* Start from initial condition, ruleset, target
* Use ruleset to decode collections in parallel. May generated many decoded case
* Use teacher-swapping with targets and ACT process
  * One fast ACT process indicates whether this is our answer to a target.
    * We do the weighted sum to get the target answer
    * We keep the old context around, though! In particular, we mix it with the correct
      target according to the act probabilities, then continue to generate.
    * This simulates teacher-forcing.
  * One slower ACT-like probability indicates whether we are done generating for this sequence

### Feedback Distillation Unit

We need to create feedback the memory unit can access

* target_difference tensor: The difference between the targets and the true value
* ruleset_length: Information about the length of the ruleset.
* sequence_difference: Information about how long the sequence was vs how long it SHOULD have been

TODO:

* How do you handle ending masks? Token masks?

## Training 


### 


* Ruleset is 
