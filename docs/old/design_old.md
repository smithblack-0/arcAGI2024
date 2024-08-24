# Overview

This is a document designed to discuss the layers and losses
of the ARC-AGI solution proposal involving tutoring and 
programming.

## Ideas and Inspiration

The basic ideas behind this are as follows. 

**Programming**

Solving the RPM task is treated as an application in summarizing important features
into a set of context known as a "Program", then decoding that program on an input
case to attempt to get a similar output case.

Notably, I use the word program for a good reason - the program is accessed in the
decoder by a set of Neural Turning Machine read-only heads. The intention with this
is that NTM can simulate a program counter.

The task thus becomes an exercise in encoding the proper rules and context, then
using it to solve novel cases.

## Terms

* program: 
  * A fixed-length tensor of embeddings that will contain instructions on how to interpret a particular example
  * Is readable by a NTM process, and likely written by an NTM process.
  * It is intended that this will contain "Rules" which can be applied to solve a particular challenge.
* example_collection:
  * A set of input | target pairs showing the pattern
  * The model will get to see both the inputs and the targets.
* holdout_collection:
  * The cases we are trying to infer. The model does not see the targets
  * Consists of input | target pairs.


## Layers

Primary model pieces are discussed here. These include the 

* ProgramEncoder
* ProgramDecoder

### Program Encoder

The program encoder is responsible for looking at a particular example collection,
and possibly holdout input cases, then should come up with a program that can solve the 
scenario.

**Premise**

The program encoder should be responsible for encoding a "program" that can be
used to evaluate the problem. This program will likely contain information like
objects present, and information on how they behave between input and output cases

**Dependencies**

* NTMSpawner: Spawns a blank program
* NTMWriter: Writes to the program.
* TransformerDecoder: A transformer decoder without a logit head. Should accept context.

**Accepts**

* prompt: An embedding tensor containing prompts, the various different category cases, and 
          other important related information.
* mask: Attention masks.

**Returns**

* command_sequence: The command sequence used to make the program
* program: The program itself.
* act_metrics: How the act process is behaving,

**Design**

Basically, we use a raw transformer, with the prompt as context, to generate
from the existing sequence the next command sequence that we think will help
solve the problem. 

Notably, it is assumed there is one program that will help solve each situation.

Each command generated is written into program memory by the NTMWriter, and
also appended to the end of the generative tensor. 

The ACT process ensures that we can generate for as long as we think we need
to program the memory. Once ACT is done, we can add the program pieces together
using the weights to get the final result. This can then be returned.

**Loss**

The program encoder primarily receives loss based on whether the intepreter 
was able to successfully solve the task at hand, whether that be generation
of the example targets or the test targets.

### ProgramDecoder

The program interpreter will attempt to apply a program to an input so as to map it
to a target output. It is multipurpose and generally can be thought of as decoding
a program.

**Dependencies**

* NTMReader
* Decoder:
  * Custom decoder with context reading provided by NTM reader.
  * Generates using BME schema.

**Accepts**

NOTE: targets and prompt are all going to be BME collections.

* program: The program tensor
* prompt: The prompt to generate from
* targets: The targets we are attempting to generate
  * Optional: Used during training for teacher-forcing.
  * When evaluating, simply do not provide targets
* temperature: The temperature for sampling
  * Optional: Used during evaluation for sampling
  * When training has not effect

**Returns**

* logits: Prediction logits.
* predictions: The sampled predictions.
* block_type: The type of block being generated. 

**Design**

Basically, we run the decoder forward from the prompt, using the program as 
attention context, and replace the cross attention layer with an NTM access 
mechanism. The positions are retained between subsequent transformer layers

## Architecture.

There are four distinct phases of operation that the model operates under
while training in order to get the best results possible. These phases are,
roughly

* ProgramEncoding: Encodes a problem case into a program
* ExampleInference: Attempts to solve the example targets given their inputs. 
* HoldoutInference: Attempts to solve the holdout targets given the inputs
* Debugging: Attempts to explain the things that were actually important

### Program Encoding

In program encoding, a problem case is presented to the program encoder
and turned into a program. The program would hopefully
contain needed information, but not too much information

The layer used for this is, naturally, the program encoder.

### Example Inference

In example inference, we see if we can solve the cases originally 
provided. In detail, we attempt to infer the targets from the example
collection given the inputs.

The layer used for this is the ProgramDecoder. It is given a 
"decode examples" prompt

A cross entropy loss can then be generated, and will contribute
to training the model. The loss is not weighted too heavily.

### Target Inference

In target inference, we attempt to solve for the holdout
targets. We use the program, and see if we can get the job done. 

The layer used for this is again the ProgramDecoder, this time
fed with the target prompt.

### Debugging

Debugging attempts to display information that shows why we got the answers that 
we did. Not all data will include a debugging chain - indeed, we generally 
will only be able to create a debugging logic chain when making synthetic data.

Debugging may also have a loss

## Synthetic data

