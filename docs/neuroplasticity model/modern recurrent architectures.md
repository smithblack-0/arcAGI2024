# Rethinking Generative Recurrent Networks for the Transformer Era: The EBRT Cell, and Training over infinite contexts with objectives to encourage neuroplasticity.

* Small blurb to talk about modern reversable recurrent models.
* We propose a strategy to design high capacity recurrent models
  for the transformer era.
* Extra emphasis is placed on designing an architecture that is highly modular 
  and easy to implement. New insights can be inserted easily.
* We also briefly highlight how one modern recurrent model, RWKV, 
  can easily be rewritten to satisfy this strategy.

# Prior work

## History of the transformer

* Attention, feedforward step
  * Parallize much easier. 
  * Better results
  * Quadratic complexity in time
* Beating the quadratic complexity
  * Linear transformer.
  * Others

## Work division of the traditional transformer cell

The traditional transformer can be viewed in terms of "spatial" and
temporal operations that view the past in order to make predictions. They
are composed of layers that perform:

* "Temporal" mixing
  * Produce effects based on what has happened in the past
  * Attention is the "traditional" mechanism
  * "Linear attention" is a rewrite of this that has the interesting property of 
     being rewrittable as a recurrent operation.
* "Channel" Mixing
  * Produces effects based on what has happened spacially
  * "Feedforward" is the traditional implementation, although others such 
    as spacial mixing (RWKV) have also been attempted

## Refactoring as a recurrent process

Recurrent transformers are an active domain of research. They are recurrent
representations of linear transformer processes that have shown some promise. 
Under these interpretations, any "Temporal" mixing becomes a recurrent
update step.

(Do literature search)

## Optimization and training of Generative Natural Language Architectures

* Generative Language architectures have been trained on a task known primarly
  on next token prediction in a pretraining event
* Usually trained on "time-parallel" model types such as the transformer. 
* Usually trained on batch widths between 1-30.

## Reversable Timestep Cells with Feedback

For the purposes of this paper a reversable timestep
cell is said to incorporate "feedback" if it is the case
that new recurrent states are not constructed from the old
using the "no-feedbck" condition. 

### Coupling feedback

One very popular reversable cell consists of:

$s_1' = s_1 + f_1(s_2, x)$
$s_2' = s_2 + f_1(s_1', x)$

Where $s_1'$, $s_1$ are recurrent state 1, and $s_2'$, $s_2$ are recurrent state 2, 
and $f_1$, $f_2$ are some sort of gates. This gate is completely numerically stable under reversal,
as any error that occurred in the forward step will be compensated by the same error occuring in the 
backwards pass. However, it has capacity issues.

### Reversable LSTM or GRU

One way around this is to try to simulate a LSTM or GRU that is timestep reversable.
But that also has issues. It tends to fail at the job if no erase gate is provided,
while if one is provided irregularities between the forward and reverse step torpedo
the process.

(MacKay et al., 2018)

# Rethinking the Recurrent Transformer Cell for Reversibility over Arbitrarily Long Token Length.

## An Introduction to the EBRT Recurrent Cell

The **Embedding-Based Reversible Timestep (EBRT)** cell introduces a novel 
mathematical framework designed to enable efficient, memory-constant training 
over nearly arbitrarily long token sequences. By leveraging a forward and 
reverse timestep mechanism, the EBRT cell achieves full reversability along the 
temporal dimension, requiring only a single token’s computational graph to be 
stored in memory at any given time. This reversability eliminates the need to 
store intermediate states, allowing the model to process sequences with 
effectively infinite context lengths.

While achieving these capabilities during evaluation is already possible in 
many architectures, the EBRT framework uniquely extends this capability to 
**training**, enabling models to process nearly infinite sequences without 
memory growth. This is achieved by reconstructing intermediate states on-the-fly 
during the reverse pass, ensuring that training remains memory-constant 
regardless of sequence length. By introducing this innovation, the EBRT 
framework unlocks the potential for training with extremely large contexts, a 
capability previously limited by memory constraints.

The EBRT framework trades computational equivalency to the infinite memory case 
for constant memory usage. The computational cost is a flat 2–3x increase due 
to the forward and reverse passes, but this trade-off unlocks pseudo-unlimited 
sequence lengths while maintaining consistent and stable training dynamics. 
Numerical rounding errors, while inevitable over very long sequences, grow 
logarithmically with sequence length. This logarithmic relationship ensures 
that precision loss remains manageable, and if necessary, can be further 
mitigated by adopting higher-precision data structures, such as float64.

A key feature of the EBRT cell is its scalability. The logarithmic relationship 
between precision loss and sequence length implies that future architectures can 
extend sequence lengths indefinitely by adapting numerical precision. For 
practical purposes, float64 already supports sequence lengths exceeding 
67 million tokens during **training**, far beyond the requirements of most 
applications. This flexibility positions the EBRT cell as a future-proof 
solution for long-sequence modeling.

In this section, we introduce the EBRT cell’s formal mathematical framework, 
including its external and internal contracts, which govern its reversable 
behavior and numerical stability. We also discuss its precision limits during 
training, providing an analysis of common floating-point formats and their 
suitability for training over extended sequences. Finally, we outline a 
practical roadmap for implementing EBRT cells using modern machine learning 
libraries, enabling researchers to leverage the benefits of infinite-length 
contexts efficiently.

### Formal Specification of the EBRT Recurrent Cell

The Embedding-Based Reversible Timestep (EBRT) recurrent cell is defined by two
levels of specification: the external contract and the internal contract. The 
external contract governs the cell's observable behavior, ensuring reversability
and compatibility with sequential processing. The internal contract describes how 
the cell manages updates to its hidden state, ensuring numerical stability over extremely
long training lengths

#### Definitions

* x_{in}, x_{out}, s_t, s_{t+1}
* f_forward, f_reverse
* g_update

#### **External Contract**

The external contract defines the EBRT cell's observable behavior through two 
core functions: the forward function, $f_{\text{forward}}$, and the reverse 
function, $f_{\text{reverse}}$. Given the previous definitions, these are 
structures as, for forward and reverse respectively:

$$
s_{t+1}, x_{\text{out}, t} = f_{\text{forward}}(s_t, x_{\text{in}, t}),
$$

$$
s_t, x_{\text{out}, t} = f_{\text{reverse}}(s_{t+1}, x_{\text{in}, t}).
$$

The forward and reverse functions, $f_{\text{forward}}$ and $f_{\text{reverse}}$, 
are defined as mathematical inverses with respect to the temporal progression 
of the hidden state. Unlike models such as RevNet (Jacobsen et al., 2018), which 
invert operations to reconstruct input activations from outputs the EBRT cell 
performs inversion across timesteps. This behaves more in analogue to 
the reversable LSTM and GRU of (MacKay et al., 2018), This means that the reverse
function reconstructs the previous hidden state $s_t$ using $s_{t+1}$ 
and $x_{\text{in}, t}$, satisfying:

$$
f_{\text{reverse}}(f_{\text{forward}}(s_t, x_{\text{in}, t})) = s_t,
$$

$$
f_{\text{forward}}(f_{\text{reverse}}(s_{t+1}, x_{\text{in}, t})) = s_{t+1}.
$$

This timestep-based inversion eliminates the need to store intermediate hidden 
states during training or inference. Instead, these states can be reconstructed 
on-the-fly using the input tensor to the cell x_in and the reverse function. 
Ultimately, this means that if a model consists exclusively of cells satisfying
the contract, this means we can perform the reverse step using only the input embeddings, 
giving us the name.

#### **Internal Contract**

To ensure numerical symmetry and stability between forward and reverse 
operations, the EBRT cell must adhere to an internal contract governing how its 
hidden state is updated. This contract includes two mathematical constraints, 
**write-first** and **write-once**, which are essential for invertibility, and 
a numerical constraint, **additive updates**, which ensures favorable stability 
properties during training. Together, these principles allow the EBRT cell to 
maintain its reversable behavior over extremely long sequences.

*Mathematical Constraints for Invertibility: Write-First and Write-Once 
Principles*  

The **write-first principle** and the **write-once principle** derive directly 
from the mathematical requirements for invertibility. In order for the EBRT 
cell to function correctly, its forward and reverse operations must yield a 
system where each hidden state can be uniquely reconstructed. This is only 
possible if every timestep’s hidden state update adheres to the following 
constraints.

First, the **write-first principle** dictates that the hidden state must be 
updated before it is read for further operations. Formally, this means that the 
output at timestep $t$ must depend on the updated hidden state $s_{t+1}$ rather 
than the original state $s_t$:

$$
x_{\text{out}, t} = k(s_{t+1}, x_{\text{in}, t}),
$$

and not:

$$
x_{\text{out}, t} = k(s_t, x_{\text{in}, t}).
$$

If this condition is violated, the reverse operation cannot uniquely determine 
$s_t$, as there would be one equation but two unknowns.

Similarly, the **write-once principle** requires that the hidden state be 
updated exactly once during each timestep and that this updated state is the one 
returned by the cell. Multiple updates within the same timestep result in 
overlapping dependencies, leading to ambiguity in the reverse operation. By 
ensuring that $s_{t+1}$ is computed with a single definitive update, the system 
guarantees a consistent mapping between timesteps. Without this constraint, it 
would again be impossible to write a function that is invertible.

These principles are non-negotiable mathematical requirements for invertibility. 
Without them, no formulation of $f_{\text{forward}}$ and $f_{\text{reverse}}$ 
can guarantee reversable behavior.


*Numerical Constraints for Stability: Additive Update*  

The **additive updates principle** is not a mathematical requirement but a 
numerical optimization that ensures the system remains stable during training. 
When processing in a batch, a masking term is introduced to account for the 
varying sequence lengths or positions of individual tokens within the batch. 
The hidden state update is computed additively as:

$$
s_{t+1} = s_t + m_t \cdot g_{\text{update}}(x_{\text{in}, t}),
$$

where $g_{\text{update}}$ processes the input embedding, $x_{\text{in}, t}$, and 
$m_t$ is a masking term that adjusts the contribution of $g_{\text{update}}$ 
based on the current batch state. This ensures that updates respect 
sequence-specific constraints within the batch while maintaining consistent 
numerical behavior.

This additive structure, augmented by the masking term, has significant 
numerical advantages, particularly over long sequences. By isolating updates to 
relevant portions of the batch, the framework avoids unnecessary precision loss 
and maintains stability during training.

It should also be noted that the additive update is numerically symmetric! If 
there is an error in the forward step, the same error is reproduced in the reverse
step, meaning training will behave exactly the same way whether running forward
or backwards, regardless of numeric errors.

#### **Precision Limits of the EBRT Cell**

Understanding the precision limits of the EBRT cell is crucial for evaluating 
its effectiveness over long training sequences. While the training process is 
inherently symmetric due to reversible updates, capacity can diminish over time 
as computations accumulate numerical rounding errors. The simplest and most 
effective way to mitigate this is to use **float64** for the memory tensor, as 
it provides effectively unlimited precision for most practical applications. 
This section explores how precision loss occurs and evaluates the limitations 
of commonly used floating-point formats.

**Quantifying the Rate of Capacity Loss**

To evaluate precision loss, we introduce the **Number of Updates at Half 
Mantissa Capacity (NUHMC)**, which measures the number of floating-point 
additions a system can perform before half of the mantissa bits are effectively 
rounded away. At this point, rounding errors may begin to become significant, 
depending on the model’s tolerance for reduced precision. NUHMC provides a 
conservative proxy for identifying when a change in data structure may be 
necessary.

The NUHMC is calculated by considering repeated additions of the same small 
value $s_{\text{update}}$ to an initial state $s_0$. Each addition accumulates 
rounding errors as the mantissa bits of $s_0$ and $s_{\text{update}}$ 
increasingly overlap. The point at which half of the mantissa bits contribute 
no meaningful precision defines the NUHMC. This analysis assumes the following 
conditions:

1. The updates are bounded such that $|s_{\text{update}}| < \epsilon$.
2. The updates are well-behaved, with their magnitudes distributed meaningfully 
   across their range, ensured by weight normalization or similar regularization 
   techniques.

In the worst-case scenario, where $|s_{\text{update}}|$ remains constant and 
maximally utilizes the representable range, the NUHMC is determined by the 
number of mantissa bits in the floating-point format. The relationship is 
exponential, with NUHMC doubling for every additional mantissa bit.

**Analysis of Common Floating-Point Formats**

Here we evaluate the NUHMC for three commonly used IEEE floating-point formats: 
float16, float32, and float64.

1. **Float16 (16-bit)**:  
   Float16 has 10 mantissa bits. By this analysis:  
   - Number of mantissa bits until half capacity: 5  
   - NUHMC: $2^5 = 32$ updates  

   After 32 updates, rounding errors may begin to become significant. Float16 
   is unsuitable for long-sequence training due to its extremely limited 
   capacity.

2. **Float32 (32-bit)**:  
   Float32 has 23 mantissa bits. By this analysis:  
   - Number of mantissa bits until half capacity: 11  
   - NUHMC: $2^{11} = 2048$ updates  

   Float32 offers better precision than float16, with rounding errors likely to 
   begin becoming significant after 2048 updates. However, for very long 
   sequences, float32 is still likely to suffer from degraded precision.

3. **Float64 (64-bit)**:  
   Float64 has 52 mantissa bits. By this analysis:  
   - Number of mantissa bits until half capacity: 26  
   - NUHMC: $2^{26} = 67,108,864$ updates  

   Float64 is effectively unlimited for practical purposes. Rounding errors 
   might begin to become significant only after over 67 million updates, making 
   float64 ideal for infinite-length sequences.
 
**Conclusion**

The NUHMC analysis highlights the limitations of float16 and float32 for 
long-sequence training. Float16 becomes unreliable almost immediately, and 
float32 begins to show precision degradation within a few thousand updates. In 
contrast, float64 offers unparalleled stability, with a conservative threshold 
of over 67 million updates before precision loss might become significant. For 
this reason, the EBRT framework should use float64 for memory tensors to 
maintain precision and reversibility over extremely long training sequences.

## Training over Pseudo-Infinite Context Length using an EBRT model.

### The Goal of Memory-Efficient Training

A computational graph is a directed acyclic graph representing the sequence of 
operations performed on data during forward and backward passes in a neural 
network. It is integral to frameworks like PyTorch and TensorFlow, where it is 
used to compute gradients for optimization. This typically involves storing 
intermediate memory states, or "activations," for the entire computation 
process across all timesteps, requiring a model memory footprint of at least 
$a \cdot N$, where $a$ is a large constant and $N$ is the number of tokens in 
the sequence being processed.

The training process implemented in this section is designed to be equivalent 
to building the full computational graph described above but requires only a 
constant memory footprint of $a$, regardless of the number of timesteps 
executed. This is achieved at the cost of doubling the training time. While 
the process does not eliminate the need for $N$ units of memory for the 
embeddings and $N$ units for random state caches, these can be cached off the 
accelerator with a minimal performance hit, reducing the memory burden 
significantly compared to storing the entire computational graph.

This approach is mathematically equivalent to executing the graph in one go, 
as in a standard PyTorch or TensorFlow model. By rebuilding the graph in the 
"reverse" step, "reinjecting" gradients from the previous step, and "scaling 
down" the accumulated losses to account for processing multiple timesteps, 
constant memory usage is achieved.

Processing one token at a time would typically result in poor training 
efficiency, as the accelerator (GPU or TPU) would remain underutilized. This 
limitation can be overcome by using exceptionally wide batch widths, which 
fully saturate the accelerator while maintaining a constant tokens-per-second 
rate regardless of sequence length. Since the EBRT process operates in constant 
memory, the traditional trade-off between batch width and sequence length is 
eliminated, allowing this "batch-parallelized" operation mode to operate extremely
efficiently.

In this section, we discuss:

* The mathematical formalization of the recurrent training process and its 
  equivalence to the forward computation process with infinite memory.
* The capability of this formulation, under infinite precision arithmetic, to 
  train over an infinite series of tokens in constant memory.
* Practical considerations that limit pure adoption of the formalism and the 
  trade-offs necessary to maintain excellent performance.
* Implementation of an EBRT training process, including pseudocode and 
  specialized PyTorch layers to simplify the integration of such mechanisms.

### Formal Description and Analysis of the EBRT Training Process and 

#### Formalization of EBRT Training

### Formalization of the EBRT Training Process

The EBRT training process requires precise handling of forward and reverse 
computations to enable memory-efficient training while maintaining gradient 
flow. Both computations explicitly depend on the model parameters, token mask, 
and random state. These additions ensure proper handling of masked tokens 
during training and guarantee reproducibility through the reverse pass.

The forward and reverse functions are, respectively:

\[
s_{t+1}, x^{(\text{out})}_{t}, \mathcal{R}_{t+1} = f_{\text{forward}}(s_t, x^{(\text{in})}_{t}, m_t, \theta, \mathcal{R}_t),
\]

\[
s_t, x^{(\text{out})}_{t} = f_{\text{reverse}}(s_{t+1}, x^{(\text{in})}_{t}, m_t, \theta, \mathcal{R}_t),
\]

Here, \( m_t \) represents the token mask at timestep \( t \), \( \theta \) is 
the set of model parameters, and \( \mathcal{R}_t \) is the random state. The 
forward computation updates the hidden state \( s_t \) and computes the output 
embedding \( x^{(\text{out})}_{t} \), along with the next random state 
\( \mathcal{R}_{t+1} \). The reverse computation reconstructs the preceding 
hidden state \( s_t \) and the output embedding \( x^{(\text{out})}_{t} \), 
ensuring proper gradient propagation for training.




## Recurrent Transformers are EBRT: Immediate applicability of Recurrent Transformer Research to the EBRT paradynm

## True Neuroplasticity Pretraining in the EBRT Paradynm: Taking Full Advantage of the Infinite Context Length

# Experiments and Experiment Proposal

## Technology demonstrator

* personal funds and resources
* Illustrated that EBRT technique of processing one token at a time does work and wide batches
  can still fully saturate the accelerant (in that cae the GPU)

## Development of a 

## RWKV rebuild/retrain as a EBRT model with extended pretraining objectives.
* Technology Demonstrator.
* Need funding for anything else.
* Need a team for anything else. 

# Extensions

* EBRT-Infinite: True infinite length training by restrictive mantissa
* EBRT-Int: Extending precision by 

# Appendix:

-
- Modifications to rebuild RWKV as a EBRT cell.
- 