# What is the transform executor?

The transform executor is responsible, perhaps unsurprisingly, for executing a transform
against an input latent state. 

It should be noted that the objective of the entire process is to get the right set of transform context
to pair with the transform executor in order to be able to generally handle the various problems that 
exist. 

# Transform Executor Contract:

## Input Contract:

The TransformExecutor will be fed the latent_targets, the transform_seed, and 
the cognative context. Lets talk about each of these groups.

latent_targets:
  * These are latent space representations of what we want to start at and then sequentially generate
  * latent_representation: (batch x collection x sequence x ...)
  * latent_mask: (batch x collection x sequence)

cognitive_context:
  * This contains seed and context information for the transform
  * seed: A place to start generating at. Shape (batch x 2*embedding)
  * context: The context we have observed. Shape (batch x N x embedding)
  * context_mask: The mask. Shape (batch x N x embedding)

## Output Contract for the TransformExecutor

The TransformExecutor agrees to provide you with the transform result,
the process costs, and the context activity scores.

The transform results are
exactly what they say on the tin: what we got back when running the transform. It
has the same shape as the input. In our application, we will discard the last element.

transform_results:
* The output from applying the transform directive
* latent_representation: Shape (batch x collection x sequence x ...)
* latent_mask: The mask. Shape (batch x collection x sequence)

The process costs represent, basically, how difficult the transform computation proved to be. 
Adaptive Computation Time is used to allow multiple steps to occur against a single input. This
has the side effect of leaving us with a nice cumulative probability score we can use to rate
how difficult the job was. Details on how that is constructed will come later.

We also collect information on how much of the context we have to access in order to get the
job done. This is tracked using a specialized version of cross attention which uses sigmoid
rather than softmax units. 

process_costs:
* cost_steps: 
  * How many ACT steps were needed. 
  * Differentiable
  * Shape (batch x collection).
* cost_context_access:
  * How many piece of context were needed to get the job done.
  * Less is better.
  * Shape (batch x collection x sequence)
  * Derived from the entropy of the attention weights distribution.

This information will be available to the cognition model, and will also influence final loss
scores.

## Logic Contract for the Transform Executor

The Transform Executor must execute certain pieces, in a certain order, in order
to produce sane results.

### Generative Process. 

The generative process begins by concatenating the seed onto the embedding dimension of the 
latent representation. This is then fed into a specialized decoder along with the context, and an ACT
process runs until all sequence dimensions are halted. Once complete, the embeddings are projected
back down to the proper size and this will form the latent_output.

Along the way, we collect two things. These are the halting probabilities in each ACT step, and the
the context access requirements in terms of the attention weights. We use the inverted cumulative probabilities
to add together the context access requirements from each layer. This gets us the cost_steps,
and cost_context. 