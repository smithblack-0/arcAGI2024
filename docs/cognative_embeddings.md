## Cognative Embedding

A cognative embedding (CE) is an embedding which consists of a considerable quantity of related information which is accessable by attention. The interface with the transformer needs consideration.

### The naive approach: Simple Embedding 

The simplest embedding might just be a constant width vector. This has several pros. 

* It is straightforward
* It involves compression to a vector, a topic that have been extensively researched
* A flavor of autoencoder, which is heavily researched, should be sufficient to encode an entire collection of inputs given long enough tensors.
* You might be able to make additional examples for free if your encoder-decoder is a VAR.

It also, however, has some limitation as well:

* A constant width vector will eventually run out of room for context as messages and grids get longer.
* Positional information will not tend to encode neatly into a single vector.

The naive approach is, make no mistake, possible. But it has its limitation - the spectre that has long been an issue, the constant-width context running out of room.

### Transformers in transformers.

One approach we might use to get around this is to represent the entire problem in terms of a sequence of transfomer embeddings, then perform attention within attention by attending to each cluster of attention embeddings.

This has the issue associated with it that you are no longer hiding the computational complexity of the section but must deal with the entire thing. Which is not desirable. 

This approach will NOT be used. However, in theory, something along these lines would be fairly ideal in terms of not losing generality.

### Linear Kernel Embeddings: Fast without loss of generality.

We are greedy. We wish to maintain access to as much information as possible, without being forced to actually perform attention with respect to all the discrete little details. Can we make that happen?

Suprisingly, yes. The kernel trick for linear transformers allows for this to occur without significant loss of generality. We encode our "Embeddings" as a set of vectors representing the attention kernels. Then we can use that in the later attention mechanism to perform rapid linear attention across an entire collection.

Due to noncomplete equity between kernel attention and standard attention, the generative attention operation will remain in standard format for now. We will not be generating extremely long sequences anyhow.

* source: https://arxiv.org/abs/2011.01136

### LKE specification

A linear kernel embedding satisfies the following properties. For each item in the embedding, the item represents a collection of numerous elements consisting of tokens, grid elements, etc in terms of matrix $M$ and a normalizer $n$ in a manner consistent with "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

This specification will form the core format used to encode information by the model, and even the expected output format from each step of the transformer.

## LKE Variational

The LKE variational format is like the above, except we are given a mean, std pair for each data point. It is used for variational autoencoding. In specific, each LKE variational tensor collection consists of:

* A MatrixMean tensor
* A MatrixStd tensor
* A NormalizerMean tensor
* A NormalizerVar tensor
* A Count of the number of entries being stored.
