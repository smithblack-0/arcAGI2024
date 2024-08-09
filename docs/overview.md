# Introduction

This contains the documentation and specifications for getting the cognition model ready to go. 
It discusses what is going on in each subunit, what is needed for operation, and why. 

## How we challenge the ARC-AGI 

Basically, the ARC-AGI case is all about finding a map with minimal data. We propose
an architecture that experiments in a latent space with various maps in a manner reminisent of
reinforcement algorithms to find what looks like an optimal map, then applies that to the test
case. We use sophisticated data extension techniques, pretraining, and data synthesis to extend the 
training set.

We also decouple the cognative engine from the actual task at hand, in the hopes that said engine
can be generalized to many different tasks.

## Levels of thought

There are basically three levels of "thought" in the model. These might best be described
as layers and levels that reflect how close to the actual data we are. If we were to make catagories,
the model has information divided up into working with raw data, a latent spae zone, and
a cognition zone. Lets go through what is going on in each zone, and how information moves
between the zones

**Latent space**

The latent space can be thought to be analogous to the environment of a reinforcement learning
algorith. It is a space in which to perform experimentation. The space has direct corrolations
between the shape of the tensor and the shape of what is being represented. ]

The latent space is tied to a particular problem being investigated, like the ARC-AGI challenge, 
a mock data prediction case, or something similar. It actually represents the environment.

**Transform Space**

Transforms, meanwhile, consist of a batch of tensors of a certain length along with the transform
mask. They will be applied in sequence. They exist in a "transform space" that can show us what
previous transforms have been constructed along with how they did.

**Cognition engine**

The cognition engine, meanwhile, tries to find an instruction and ruleset that solves the case. It
recieves feedback based on how well a given rule does in the form of losses, feedback, and views. It
is not heavily tied to the latent representation, and in fact communicates primarily by querying the representation
in a shape-independent manner. 

The cognition engine can be thought of driving the problem forward, by looking at the various transform
options and receiving feedback on what is doing well.

## Communication between engine and latent space

Communication between the engine and latent space is highly controlled in order to ensure the
process remains highly modular. The hope is the same engine can be trained or pretrained on many
different tasks.

However, one primary thing that should be kept in mind is that communication between the latent space
and the cognition engine should be fairly dimensionless. 