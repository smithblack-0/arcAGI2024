# Introduction

The latent map is a layer that is capable of taking in and transforming a latent space
from one configuration to another. It is a very important part of a model, and may
end up configured differently depending on the task being targeted.

This is a specification file for it. We will specify a contract for our latent
mapping process to fufill.

## Class: Transform_Planner

This, basically, produces a sequence of tensors we can call "transforms" that we will
want to apply in sequence to the problem to get the various cases.

Command, LSTM -> cases, retention_probability, done_probability
cases, knowledge_context -> transforms.
    

### Premise

* We will want to support long sequences for pretraining purposes
* We need to be creating standardized 

## Class: Latent_Map

### Premise

We are assuming the task in ARC-AGI can be represented in terms of trying
to find the correct map.

### Depends

* transform_planner: Plans out a transform sequence.   
  * It will produce the same context which is shared between all collections
* transform_committer:
  * 

### Accepts:

* latent_start: The starting latent tensor. Shape varies depending on application
* command: A per-batch command to execute. 
  * Shape: (batch x embedding_dim)
* thoughts: A tensor of ideas to associate
  * Shape (batch x N x embedding_dim)
  * N is arbitrary

### Returns:

* latent_response: The resulting latent tensor. Shape varies
* transform_cost: The cost of the latent transform.

## Design

There are basically two parts to this. These are to extract the relevant thoughts, then 
