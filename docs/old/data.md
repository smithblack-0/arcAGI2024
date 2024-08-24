# Overview

We are going to need significant quantities of relevant, synthetic data in order
to train the model. 

## Pretraining

There are two things that we need to learn over the course of pretraining. 

One of these things is how the "environment" works. That the edges of 
a intgrid are edges and can be relevant to a problem. That sections
can be shifted around. And similar processes.

The other very important thing to work on is logic. The ability to look 
at a task, hopefully a novel task, and fit a pattern to it.

Pretraining will be peformed using the RAVEN challenge and a custom transforms
implementation designed to teach the model about some of the interesting
complications it might face.

Together, these will hopefully both teach it logic, and get it ready to 
face the challenge

### RAVEN

The RAVEN dataset will be used for pretraining purposes - however, with a few 
minor modifications.

* 

### Uncontrolled Transform Pretraining

The UTP task is designed to get the model used to understanding some of the 
unique quirks of the ARC-AGI mechanism.



## Identified Requirements

**Boolean Math**

* and, or, xor, nor, nand, etc.
* physical
* controlled

**Matrix controls**

You behave differently depending on the elements of a matrix

**Physics**

Including collision with edge, and collision with entity

**Repeat Pattern**

* between start and end points
* forever

**Tiling**

Restoration

**Cropping**

Controlled crop.