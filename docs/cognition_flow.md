# Overview

This is a file to help visualize how commands are issued, processed, and feedback 
integrated in order to get the job done. It is an overview on the integration 
between the various pieces

## Objective

The objective is to find the best transform directive to fit the situation. The model should
try possibilities, observe their result, consider the feedback, and revise it's approach.

## Generative context

We assume we have as data a sequence of examples that consist of various steps of a process.

# Model flow

## Generating a transform directive

We start by assuming we have a running cognitive engine, and working cognitive_command and
context tensors. The cognitive tensors are used to create a transform_directive - a series of 
transforms to apply - based on the command and the context. 

## Executing a transform directive.

The transform directive is executed by the transform engine.
It uses:

* transform_directive: What to do
* starting_space: The starting latent space

* target_space: The target space to generate
* target_max_steps: The maximum number of transforms to execute before reaching the target


**OPEN QUESTIONS**

* what is a good architecture for this?
* How do we handle teacher forcing?
* How do we wean ourselves off teacher forcing?

## Feedback based on the directive

Feedback can then be gathered. We have the synthetic latent space, and the example latent
space. We use

* feedback queries
* synthetic space
* example space

And run this through the feedback engine. We compare the two to produce meaningful feedback to
answer the queries. This includes details such as loss, and other important details

## Integration into the cognition engine

The cognition engine now attends to this feedback and proceeds to the next step.