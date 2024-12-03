# A discussion of efficient parallelization of infinite-length-context recurrent cells

## Abstract
- We define the concept of the "timestep reversable checkpointing" recurrent cell for highly
  efficient, parallel, training of recurrent models suitable for usage in the
  emerging recurrent alternatives to transformers.
- We discuss the format models must comply with in order to be "token reversable checkpointing"
  compatible, and the training infrastructure that should be used. This includes timestep 
  gradient accumulation and the forward/reverse pass.
- We create a python training infrastructure to train such 
  a model that can be distributed and used to elegantly train
  these models
- We discuss existing experimental results and issues.

### Literature review

This is to be implemented

- Define the idea of "checkpointing"
- Define and disuss previous attempts of reversibility
- Discuss some of the pitfalls of recurrent model training, and the 
  issues they have with efficient parallelization due to memory exhaustion with 
  wide batches.
- 

## Concept
### What is a timestep reversable checkpointing model?

A token reversable checkpointing model is one that pledges to process a single token timestep
in a batch as a recurrent cell at a time, in a manner such that the current input and the 
next hidden state can be used to reverse this process producing the last hidden input,
and thus we only need to have enough memory to hold the graph to process a singular batch of
tokens at a time.

The reversal process occurs over carefully constructed states with
very carefully constructed algorithms, at which point the forward 
input is recovered. 

A high level of training efficiency is maintained by using extremely wide batch widths.
This ensures that the model remains a highly parallelizable structure that will completely
saturate the accelerant when training. Careful attention to memory is required to avoid 
causing leaks between timesteps. This gets rid of the traditional issue
with wide batch sizes - you quickly run out of memory.

Experimental results included using a batch width of 800 tokens were able to partially
train a technology demonstrator over an effectively unlimited token depth. 

### Compelling reasons to pursue this line of research

- RWKV is compatible with this architecture if the logic is rebuilt

### What is required for a model to be timestep checkpoint reversable?

Consider the layer f_n obeying:

$$x_{(t, n)}, s_{(t, n)} = f_n(x_{(t, n-1)}, s_{(t-1, n))$$

Where x_{(t, n-1)} are the inputs from the last layer which can 
consist of any number of inputs, and s_{(t, n)} consists of 
a single recurrent state relevant to this model. Such a layer
is token checkpoint reversable if and only if we can define 
a inverse function g_n such that

$$s_{t-1, n} = g_n(x_{(t, n-1), s_{(t, n))$$

Allowing recovery of the prior state from the current one.
This condition can be satisfied so long as the model is processed
one timestep at a time, allowing x_{(t, n-1)} to be known. We then
have a 'forward' pass, involving applying f_n and which computes the hidden
state, a 'reverse' pass, applying g_n, then f_n, and then use the reconstructed graph for the loss. This
allows us to maintain state in constant time, allowing essentially unlimited
training length.

### Numeric stability of reconstruction

In order for the reverse pass to be useful, 
it must be numerically equivalent to the forward pass.
Random numerical error will cause divergence. It is important
to quantify how severe that can get before issues will 
develop.

It is the case that a linear gate is sensitive to error
primarily over the typical input domain. Lets define this 
as the domain from the magnitude of the bottom 80% of inptus
to the top 80% of inputs

Consider f'_n, where that only returns
the next state. We consider the computation to be 
numerically stable so long as it will be fed into a 
gate whose error depends principly on absolute value -
which includes the linear gate - 
and that for percent error epsilon on the numeric reconstruction
it is the case that

s'_{(t, n-1)} = g_n(x_{(t, n-1)}, s_{(t, n)})*(1+epsilon)
error = |s_{(t,n)} - f_n(x_{(t, n-1)}, s'_{(t, n-1)})|
percent_error = error/gate_domain 
percent_error < sigma over all parameters and expected inputs.

### Training and implementation

For the sake of implementation, it is most straightforward
to modify the training process as follows

- The model has a .forward step accepting the input token x_{t}
  and the previous stack of recurrent state s_{t-1}, returning
  s_{t}, and the output logit l_t
- The model has a .reverse process accepting state s_{t} and 
   x_{t} and producing s_{t-1}, s_t, and l_t, with s_{t-1}
   setup to capture gradients

Then:

- The training process performs a forward pass, finding
  out what the final s_{t} state will be. 
- Gradient reinjection torch endpoints inject the gradients
  on the state allowing elegant construction of a graph.
- The training process runs a reverse process, takes a loss
  and accumulates gradients without taking an optim step. It
  also integrates gradients from the previous timestep.
- This repeats until all losses have been taken.

Additionally:

- Numeric metrics continue to be measured

### Memory and time analysis 

- Only need during the forward pass to hold the batch of 
  current tokens and the previous state, and enough memory
  to build the graph
- Only need during the backwards pass to hold the gradients 
  on the next state s_{t}, that state, and enough memory
  to rebuild the graph.
- Computation time is O(N).
- Timestep processing rate is O(1): Constant. Token rate is constant
  with respect to time.

## Concrete experiments

### Infinite length LSTM

- Show how the architecture allows the training of 
  an infinite length LSTM that can be provided 
  any number of tokens to process so long as you can
  fit the token collection in memory.
- Show how the memory usage is indeed constant, and the 
  tokens per second is about O(1) with respect to sequence 
  length
- Quantify and discuss rate of numeric drift and discuss
  checkpointing strategies used to offset this. 

## Extensions and future work


### Rewrite RWKV to be compatible

- Rewrite the RWKV architecture to use the existing pretrained
  parameters with equivalent logic but structured as a recurrent
  cell, allowing a capable transformer-peer model to train over
  infinite context length