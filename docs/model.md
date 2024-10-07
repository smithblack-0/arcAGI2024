# Introduction

* Section talks about the specialized model used
* Model is designed to hopefully be computationally efficient and generalize well
* Generalization is encouraged by allowing the reuse of computation features using
  the processing stack.
* Model is driven by a recurrent transformer architecture.

# Core components

* Linear recurrent transformer
* Memory bank for recurrent transformer
* Processing stack
* Output accumulators

## Processing stack

This is one of the core innovations of the model, and the thing that
is supposed to provide good generalization.

* Computations are performed within a differentiable neural stack
* Both the memories, and the embeddings, are maintaining in such stacks. 
  Updates can propagate up, down, or noop
* Output is accumuated content off the end of the stack. 
  * Accumulated ACT style
  * Based on how much probability was dropped off the edge of the stack.
* Basically, a process of getting a superposition from the stack, processing
  that superposition, and scattering updates back again is done over and over.
  This repeats until the stack is empty of probability.
* When a memories, embeddings combination pops out the stack, it is accumulated based on the 
  probability.
* Outermost process

### Pseudocode:

Note that there is significant custom complexity hidden in stack update. This is NOT just a normal
stack, but instead will tend to add+normalize when destacking. 

```python
# Setup outputs

memory_results = make_memory_accumulator(memories)
embedding_results=  make_embedding_accumulator(memories)

# Setup stack
memory_stack = setup_memory_stack(memories)
embedding_stack = setup_embedding_stack(embedding)

While not (memory_stack.done() and embedding_stack.done()):
	# Get superposition for differentiable stack
	memories = memory_stack.get(stack_probabilities, memory_stack)
	embedding = embedding_stack.get(stack_probabilities, embedding_stack)
	
	# Compute further features.
	embedding, memories = compute_results(embedding, memories)
	stack_actions = compute_stack_actions() #One of enqueue, noop, dequeue
	
	# Update stacks
	memory_update, update_probs = memory_stack.update(stack_actions, memories)
	embedding_update, _ = embedding_stack.update(stack_actions, embedding)
	
	# Update accumualtors
	memory_results = accumulate(memory_results, memory_update, update_probs)
	embedding_results = accumulate(embedding_results, embedding_update, update_probs)
	
return embedding_results, memory_results
```

## Bottlenecking and computation feedforward

The embeddings are typically going to be, or be projected into, a very large shape. 
Computational cost is, however, kept under control using a bottlenecked embedding when 
the transformer itself is thinking. A feedforward mechanism allows the model to arrange
its parameters to be fed into the transformer. This helps prevent excessive use of the stack.

* There exists large embeddings that flow through the model
* A smaller projection is created to actually be fed into the core transformer. The output will
  be the same size. 
* The concatenation of the result and the large embeddings is fed into a feedforward mechanism
* This is then add+layernorm'ed with the original large embeddings.


## Linear transformer and memory banking

The linear transformer we use is based on the fast transformer torch library. Importantly, the
memories are banked, not defined per layer. What this basically means is that part of the transformer's
job is to select using probabilities from among N "memory banks" to actually perform linear attention
with, again producing a superposition. The results can then have their superposition reversed.

Memory Banking:

* The memory bank consists of N linear transformer "memory" units that could, if extracted, be attended to. It
  significantly expands the storage capacity of the model and helps get around the whole linear transformer
  loses quality eventually issue by allowing memory specialization
* When preparing to perform a linear attention, logits are predicted, and the memory space is sampled from. Then,
  we return a superposition of these samples. This will be the transformer memories that can actually be used.
* When getting output memories, we figure out what the update to the memory was, and again create logits.
  We get probabilities. We again store based on those probabilities

Memory Decay:

* It should be mentioned that the existing memories banks will very slightly decay when written to. 
* The model can choose what the decay constant is. Decay constant is run through a sigmoid
* This means the model can choose to emphasize faster decay for short range data, or slower decay for long term storage
* Reading from memories does not produce decay. Only writing.

Transformer:

* Transformer consists of the specialized banked attention with decay, followed by feedforward
* They will be casual 
* Several layers of these can exist. 
