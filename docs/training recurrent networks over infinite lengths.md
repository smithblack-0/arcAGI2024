# Infinite Context Widths at Minimal Implementation Pain by Layer-Local Reversable Models

## Abstract
- We define the concept of the "Layer-Local Reversable Model" consisting of 
  layer local reversable layers for highly
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


## 1. Introduction: Why Layer-Level Reversible Models for Infinite Context Widths?

### Recurrent Models vs. Transformers: The Historical Bottleneck
Recurrent architectures, such as RNNs, have historically struggled against 
transformers due to two primary limitations:

1. **Sequential Dependency:** Recurrent models process one token at a time 
   in strict sequence, requiring the completion of one token before the 
   next. This limits parallelism and prevents GPUs from being fully utilized.

2. **Memory Constraints:** Large batch sizes are needed to maximize GPU 
   utilization, but recurrent models quickly exhaust memory when processing 
   long sequences with wide batches.

Transformers avoid these bottlenecks by parallelizing computations across 
all tokens in a sequence. However, this comes at the cost of high memory 
usage and computational expense, particularly for long contexts.

### The Key Insight: Layer-Level Reversibility for Infinite Contexts

Layer-level reversible models address these bottlenecks by introducing a 
**simple yet powerful framework** for memory-efficient training. The central 
innovation is that **reversibility can be implemented locally at each layer** 
without altering the forward-pass logic, making it straightforward to adapt 
existing recurrent architectures.

This approach:
- Processes **batches** of tokens one timestep at a time while maintaining 
  memory efficiency. These batches tend to be enormous to saturate accelerators.
- Associates each layer’s forward computation with an **inverse function** 
  that reconstructs prior states from current states, where appropriate.
- Avoids storing intermediate states during the forward pass, instead 
  reconstructing them dynamically during the reverse pass.

By leveraging layer-local reversibility, the model processes tokens 
sequentially while simulating the efficiency of parallelized gradient 
accumulation.

### A Brief Overview of the Training process.

This architecture introduces three key steps:

1. **Forward Pass:** Processes a wide batch of tokens, one timestep at a time, 
   starting at the first timestep and progressing sequentially forward.  
2. **Reverse Pass:** Reconstructs the computational graph by applying an 
   inverse function at each layer to recover prior recurrent states, working 
   backward from the last timestep to the first. The forward pass is then 
   rerun to provide the necessary autograd graph for backpropagation.  
3. **Backward Pass:** Computes gradients using standard autograd on the 
   reconstructed graph. Losses are accumulated across the entire context, 
   ensuring memory usage remains constant throughout.

### Why This Matters
Layer-level reversible models breathe new life into recurrent architectures 
by addressing their historical limitations:
- Memory constraints are alleviated, allowing efficient scaling to long 
  contexts without requiring specialized hardware or model sharding.
- GPUs are fully utilized, enabling parallelism through wide batch 
  processing.
- With effectively infinite training context lengths, this approach opens 
  new possibilities for tasks requiring long-term context, such as natural 
  language modeling, time-series analysis, and reinforcement learning.
- By significantly reducing memory requirements, larger models can be 
  trained without exceeding GPU memory limits and requiring sharding, making this method a 
  cost-effective and scalable alternative to transformers.
- Implementing a Layer-Level Reversible architecture requires minimal 
  changes to existing recurrent architectures. Since both the forward and 
  reverse passes operate in the same direction—layer by layer—the adaptation 
  process is straightforward and practical for a wide range of models.

## Mathematical Foundations: Building gates inherently compatible using the No-Feedback condition


It is possible to engineer cells that are directly reversable, rather than
are reversed by a mixture operation that can be undone as in the above. 
We discuss what the coniditons are for this to happen.

### Designing layers for Layer-Level Reversability.

* No-Feedback condition makes it possible to design inverses that operate under no layer condition
* No-Channel-Crosstalk condition 
* 
### **Definition of The No-Feedback Condition**

The No-Feedback Condition states that, during the computation of recurrent 
updates at any layer $n$ and timestep $t$, the hidden state update 
function must compute control gates and intermediary outputs independently of 
the recurrent state being updated. Consider all layers whose consumption 
of recurrent state can be written as:

* $s_{(t, n)} = f_n(x_{(t, n-1)}, s_{(t-1, n)})$

where:

* $s_{(t, n)}$: The new recurrent state after this computation.  
* $x_{(t, n-1)}$: Any number of input tensors depending on previous computations, such as control gates.  
* $x_{(t, n)}$: Outputs from this computation.  

The No-Feedback Condition is satisfied if and only if it is the case that for all layers such a function 
can be developed with the property that:

* $x_{(t, n-1)}$ has **absolutely no dependency** on $s_{(t-1, n)}$.  

It should be clarified that it is **not required** that the entire layer, including its outputs, 
must satisfy this condition. Rather, it is only necessary that the state that will be recurrently 
updated satisfies this property. Without this condition, the feedback loop introduces dependencies 
that cannot be resolved using only the inputs from the forward pass, thereby breaking the ability 
to perform Layer-Level Reversibility.

In practice this is not as severely a limiting condition as it might first appear. 
While it is true that several standard recurrent architectures - such as LSTM or GRU - 
are excluded by this process, others such as many Linear Transformer recurrent formulations
are not. 


### **Numerical Restabilization of Linear Operations**



### **Channel Independence Condition**

The **Channel Independence Condition** ensures that computations within a 
recurrent mechanism are isolated across channels of the recurrent state. In effect,
we assert that all logic within the gate is performed as elementwise operations
only.

While this is not, technically, a requirement for reversability it has such a drastic

* Shao

#### **Definition of Channel Independence Condition
For a recurrent state tensor $s_{(t-1, n)} \in \mathbb{R}^{C \times D}$, where 
$C$ represents the number of channels and $D$ represents additional dimensions 
(e.g., spatial or feature dimensions), the **Channel Independence Condition** 
is satisfied if, for all $l \in \{1, \dots, C\}$:

\[
s_{(t, n)}[l] = f_n(x_{(t, n-1)}, s_{(t-1, n)}[l]; \omega_n)
\]

where:

- $s_{(t-1, n)}[l]$: The input from channel $l$ of the recurrent state at the 
  previous timestep.
- $s_{(t, n)}[l]$: The updated state for channel $l$ at the current timestep.
- $f_n$: A function operating independently on each channel, with:
  - $x_{(t, n-1)}$: Inputs or control gates broadcastable to the recurrent 
    state tensor shape.
  - $\omega_n$: Layer-specific parameters, applied identically to all channels.

The **Channel Independence Condition** ensures that:

\[
\frac{\partial s_{(t, n)}[l]}{\partial s_{(t-1, n)}[k]} = 0 \quad \forall l \neq k
\]

This implies that channel $l$ of the recurrent state at timestep $t$ depends 
only on channel $l$ at timestep $t-1$, and not on any other channels.

#### **Implications**
- The recurrent state update is fully parallelizable across channels, making 
  computations efficient and easier to reverse.
- This condition eliminates cross-channel dependencies, simplifying the 
  reconstruction of previous states during the reverse pass.

### Directly usable recurrent cells 

Any 

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