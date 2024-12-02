# A proposal to rebuild RWKV for infinite length training

## Abstract
- We define the concept of the "token reversable checkpointing" recurrent cell for highly
  efficient, parallel, training of recurrent models in competition to transformers over infinite 
  training sequence
- We discuss the format models must comply with in order to be "token reversable checkpointing"
  compatible, and the training infrastructure that should be used. This includes timestep 
  gradient accumulation and the forward/reverse pass.

## The concept of token reversible checkpointing of recurrent cells

### What is a token reversable checkpointing model?

A token reversable checkpointing model is one that pledges to process a single token timestep
in a batch as a recurrent cell at a time, in a manner such that the current input and the 
next hidden state can be used to reverse this process producing the last hidden input,
and thus we only need to have enough memory to hold the graph to process a singular batch of
tokens at a time.

A high level of training efficiency is maintained by using extremely wide batch widths.
This ensures that the model remains a highly parallelizable structure that will completely
saturate the accelerant when training. Careful attention to memory is required to avoid 
causing leaks between timesteps.

Experimental results included using a batch width of 800 tokens were able to partially
train a technology demonstrator over an effectively unlimited token depth. 

### What is required for a model to be token reversable?


- Some recurrent cells can be written so that they are 'token reversible' for hidden states
- This means that you can reconstruct the hidden state from the last timestep given the current timestep
  and the input token.
- Once these hidden states are reconstructed, it of course becomes possible to build a graph to process
  this token. 
- Manual reinjection of gradients can then be used to perform a forward pass, finding the initial
  activation, a reverse pass, reconstructing those activations, then a backwards pass to use the
  reconstruction. Currently, I am doing this with a special trainer class. However, further research
  might be able to integrate this directly into torch's autograd graph by using pointers to common
  memory.
- The lack of parallelism is now dealt with quite cleverly. Use a much larger batch width than normal.
  Normally, this would be an issue. But since you only have to hold the graph for a single token in
  memory at a time. 
- Experimental results with a very simple technology demonstrator showed rates of 1,000 tokens/sec with
  with a batch size of 800 on a single L4 GPU. We operate in a batch parallel mode.

## Requirement for token checkpoint reversibility

- To be stable for training, the forward and inverse function must satisfy the "round trip stability" condition
- Generally, you want to use one tensor to make most of the gates used in your process in a single step
- Generally, you will want training to occur with magnitudes much higher than where error conditions happen.
- Anytime you have a tensor that is used to make a gate, you are generally going to need to pass that tensor
  on as additional hidden state. 
- Tensors should be inverse reversable, though need not be perfectly numerically stable in and of themselves.
  This means that if you have forward step y=f(x, others), reverse step x =g(y, others), with others known,
  it should be the case that in no location does (x-f(g(x, others), others)).abs() explode for parameters in
  others. 
- An interesting way to evaluate this is to evaluate the behavior under a percent error. A suprising variety
  of equations end up with a stable error. For instance, an interpolation examination shows stability when
  dividing by zero in the absolute domain where it matters. Keep in mind numeric noise will be in proportion
  to percent error.
    - f(x) = y =x_{t}*u + (1-u)*x_{t-1}
    - g(x) = (y-x_{t}*u)/(1-u)
    - percent error: 1 + epsilon
    - error = abs(x_t - f(g(x_t)*percent_error))
    - lim u->1, abs(x_t - f(g(x_t)*percent_error))=0
  This despite the fact that g(x) will trend towards some flavor of infinity.
- In general, if a computation possesses this property, it will be stable. One only has 
  to guard against numeric issues such as dividing by zero. In the above, we could easily 
  accommodate that by:
  - f(x) = y = { x_{t}*u + (1-u)*x_{t-1}, abs(1-u) > epsilon,
                 x_{t}*u, abs(1-u) < epsilon
                }
## The compatibility of the RWKV model with this numerically stabilized cell

The RWKV transformer is compatible, with minor rewriting, with this mode of training.
In fact, the compatibility is so strong that by rewriting the logic to use the parameters
in their recursive mode, it is possible to use an existing pretrained parameter set for
such a task.

### Handling the token shift condition.

The main challenge that the RWKV architecture presents to creating a reversable cell
structure is the extensive existence of the token mixing mechanism. At some point in 
every architecture up to at least finch, a linear interpolation occurs. This behaves as,
with u a trainable vector

- lerp(x_t, x_{t-1}, u)= f(x) = y =x_{t}*u + (1-u)*x_{t-1}

The reason this is a challenge, of course, is that while x_{t} will be known from the 
immediately proceeding calculation within the cell, x_{t-1} will not. So we need to be able
to reverse this. To do this, let us suppose we keep additional hidden state each iteration - 
the lerp activation. Then we have:

lerp(x_t, x_{t-1}, u) = linear_mix = y =x_{t}*u + (1-u)*x_{t-1}

From this, we can construct a reversable equation:

lerp_inverse(x_t, linear_mix, u) = x_{t-1} = (linear_mix- x_{t)*u)/(1-u)

Which can be show to satisfy the round trip error stability condition:

abs(x - lerp(x_t, lerp_inverse(x_t, linear_mix, u)*(1+epsilon), u)) < alpha

Thus allowing it to be used during training. Thus anytime we have a token shift process, we
simply need to incude the additional linear mix hidden state to enable reversable computations.
This comes as no additional computation cost, as we needed it anyhow, and at a finite memory
cost.

### Rebuilding as a recurrent cell

To make this compatible with a trainer, rebuild it so that 
the model operates in the recurrent mode. And return the 
extra information.

Then, when performing the reverse pass, use the extra
information to recover the original state, allowing 
construction of the original graph.

