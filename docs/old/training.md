# Data Pieces

* Augmented training data


# Model 

## Layers

Primary Layers

### ProgramProposer

The program proposer proposes novel "program" tensors that are suppose to encode the
rules under consideration. They are feature found only within the tutor. 

Internally, it consists of a transformer terminating in a VAR. It recieves teacher-forcing
feedback

### ProgramEncoder

The program encoder is designed to encode an inputs|targets collection of information
into a "program" tensor that can be executed later on in order to solve a particular
situation.

It is ultimately one of the things we actually are trying to train. It needs to cross
reference the various pieces of information it is fed to divine general rules that can
be used to solve unseen inputs.

### ProgramExecutor

The program executor has the responsibility of taking a batch of input conditions along
with a program and executing that program to attempt to get a target output set. It 
performs the "mapping" functionality.

### ProgramDecoder

The program decoder has the purpose of taking a program and attempting to decode from it
an input collections that could be utilized to illustrate the rules stored in the program.

## Entities

### RESTRAINT

The restraint is a fairly standard autoencoder. It performs the naive form of training
one would expect when doing an ML project - simply training against the data. However,
it contains layers that are shared by other entities and thus provides loss feedback
that also influences how other entities are allowed to operate.

It will be fed only with data from the actual training set - hopefully data 
that has been augmented by various transformations.

**Details**

Actions:

* Encode_Program
* Generate_Program
* Decode_Inputs
* Execute_Program

Loss:

* loss_program_generation.
* loss_example_input_reconstruction.
* loss_example_targets.
* loss_holdout_targets.

### SOLVER

This entity tries its best to solve a particular problem case. It takes 
in a input|target collection along with a input case to solve. It is instructed
to find the program it thinks will best solve the target case. Once the program has
been encoded, it will attempt to execute against the input.

It receives loss based on how well it did.

### TUTOR

The tutor is an interesting mechanism indeed. It is responsible for generating
cases that are independent of the training dataset, but indeed are hopefully related. 

The tutor is requested to produce a batch of examples from random noise - it does this
by first proposing a program (hopefully including some complex rules) then taking that 
program and decoding to inputs, and finally taking the program and inputs and 
turning them into outputs. 

**Loss specifics**

The tutor has an interesting loss mechanism. Specifically, the tutor aims to maintain
a certain success rate in the examples generated for the SOLVER. To be more specific, if you
design a success metric for a batch, S(batch), that returns the percentage of solutions
that were successfully found in the batch, and you let $L_{solver}$ be the loss for the solver,
the tutor's loss is along the lines of

* $Loss= f(S(batch), target)*L_{solver}$

Where it is the cae that f(S(batch), target) will be negative when S(batch) > target and
positive when less. This encourages the tutor to cooperate with the SOLVER when the problems
are too challenging, but to make them harder if the SOLVER is getting them mostly right. The 
exact details of f are yet to be firmly nailed down, but a difference would be a good start.

**Hallucination restraint 1**

The RESTRAINT layer acts as one level of hallucination restraint. 


**Question marks**

The tutor is the part of the project with the most question marks around it, and may need
additional RLHF training in order to ensure that the examples it is generated are sane. 

In specific, the tutor may tend to hallucinate heavily, and wander over to offering examples
that have little to do with the solution cases. This would in turn lead the parameters of the 
solver to learn rules that do not actually matter. 