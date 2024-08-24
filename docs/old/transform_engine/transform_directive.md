


## What is it actually?

What a transform directive actually is is two tensors. These are the *instructions* tensor,
and the *result_key* tensor.

**instruction tensor**


* (batch x instructions x embedding)

This tensor is like a sequence of instructions that are being issued. Think along the lines
of a CPU instruction pipeline. Complicating things, teacher forcing and other concerns means
that there is one instruction pipeline for each location in the sequence. Multiple instructions
can be executed between sequence locations.

It has shape:

* (batch x sequence x instruction x embedding)

Notably, when operating in learning mode, each sequence entry can be teacher-forced.

**result_weights**

The result_weights tensor plays an extremely important role in ensuring all processes
remain differentiable. It is basically a combination of probabilistic tensors in a particular
manner.

The process of Adaptive Computation Time works by performing a weighted sum using probabilities
to allow varying levels of computation for a task. That is basically what we do here. The 
instruction results can have a weighted sum performed against the result weights in order
to put together a tensor that only has shape (batch x ... x sequence x ...). 

The tensor, however, does NOT just contain probabilities. Rather, each instruction dimension
will contain only positive entries, that will sum up to be <= 1.0. That is, they contain
a depressed probability distribution.

This also acts as a mask of sorts. 

## How does this result in differentiability?

Basically, if we configure our probabilities right we can ensure there are gradients to encourage

