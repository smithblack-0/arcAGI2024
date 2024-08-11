# Generating a Transform Directive

In order for the model to work, we need to create a transform directive. Lets go
over how we will make this happen, in the CreateTransformDirective layer.

## Input Contract

The CreateTransformDirective layer will accept a start_encoding and a context tensor. 

* start_encoding: 
  * This is where the internal generative processes start. Starting state for lstm, transformer, etc.
  * Expected Shape: (batch x embedding)
* context:
  * This is additional context used during the generative process. 
  * Expected Shape: (batch x N x embedding).
  * We do not get to decide N. Use attention.
* force_length: If provided, forces generation to continue until the sequence length is a particular
                length. Used primarily during training.

## Output Contract

The CreateTransformDirective layer pledges to produce outputs consisting of the 
transform_directive, the sequence_length, and the computation_costs. These are respectively
derived internally and are differentiable quantities.

* transform_directive:
    * See transform directive
    * instructions: shape (batch x sequence x instruction x embedding)
    * return_weights: shape (batch x sequence x instructions)
    * sequence_mask: shape (batch x sequence x instructions)

We also return:

* return_probabilities: 
  * Internal probabilities tensor along instruction dimension.
  * Shape (batch x sequence x instruction)
  * Add up to one along instruction dimension
* sequence_probabilities
  * Internal probabilities tensor along sequence dimension
  * Shape (batch x sequence )
  * Adds up to one along sequence dimension

These will be used both as metrics, and to compute some additional losses. See
sequence_losses for details.

## Logic Contract

Our logic contract is as follows. 

**Generate sequence probabilities and sequence seeds**

We generate the sequence probabilities and sequence seeds by means of a decoder process. This
is generated with a starting input of start_encoding and with access to the context - presumably, 
some form of attention. During each step, we predict sequence_probabilities and stop when all are
saturated and we are above the forcing limit.

We end up with tensors

* sequence_probabilities: (batch x sequence)
* output_seed_embeddings: (batch x sequence x embedding)

**Decode seed embeddings into instructions**

Each output seed embedding conceptually represents a latent embedding at one of the target intermediate
states. The next step is to decode that point into a sequence of instructions we can execute and put together
to pull off the actual task of converting from one state to another.

We run each seed through a decoder autorecursively using ACT-like probability processes to generate
return probabilities along the way. We can provide the context again as extra information during
this process. This is done in parallel for all sequence channels. As before, we use return_probability
exhaustion to tell when all dimensions are over, or alternatively force generation to a certain length.

We will now have a sequence of instructions and an associated return_probabilities tensor:

* instructions: (batch x sequence x instruction x embedding)
* return_probabilities: (batch x sequence x instruction)

**Compute the return weights**

The return weights can now be computed. They are made in a very simple manner.
We multiply the sequence_probabilities tensor by the return_probabilities tensor,
broadcasting the last dimension of the sequence tensor.

** Returns**

We can now return the pieces

## Benefits and Reasons

The architecture is designed in this particular manner for some very important reasons. Of particular
interest is the return_weights tensor, as this ends up driving almost all the length related gradient
feedback processes.

The return_weights tensor is used to weight the interpreted instructions, which by that point has become
a collection of latent space tensors. These weights operate in an interesting manner wherein irrelevant entries
beyond the length of the sequence are not masked, but nullified by multiplication by zero. Furthermore, the
action which produced this multiplication by zero will continue to leave a gradient trail in every element
of the embedding, whether sequence or instruction related. This means if we want to make the sequence longer,
we easily can.

For similar reasons, a sequence which is too long is just as easy to adjust.

It is also possible to generate and interpret a loss based on the length of the generated
sequence, as seen in the sequence loss section.