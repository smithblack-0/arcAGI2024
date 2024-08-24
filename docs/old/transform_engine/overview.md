# Overview

The transform engine consists of three broad parts

* instruction_generator: Generates a sequence of context instructions to follow
* transform_decoder: Uses context and the instruction generator to decode instructions
* instruction penalty: 
  * We want as short an instruction sequence as is possible while still getting the job done.

## Premise

We are operating on the premise of occam's razor, basically. We want to create an instruction
generator that returns instructions that would get the job done on the requested sequence,
while ALSO minimizing the amount of instruction context that is required to get the job done.

Since the context is generated from the entire example sequence, we presume we would be gathering
the important features that are common.

## Operation

The generated instructions act as additional information for a transformer decoder's operation,
and are fed in one at a time. The probability keys are turned into actual probabilities, and used
to perform Adaptive Computation Time with the decoder output.

