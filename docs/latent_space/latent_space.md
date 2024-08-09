# Latent space collection

The entire environmental control structure of the architecture revolves around building
and manipulationg latent space collections. It is thus probably a good idea to talk about them.

One trick that is used by Tesla in order to allow their cars to be trained effectively
is to create a virtual representation of the world that the AI interacts with. All of the
cameras feed information into AI that creates a "vector space" model of where walls are in
terms of points around them. The AI then interacts with that model. This is our version of that.

## What is a latent collection?

A *Latent Collection* is a group of information that has been encoded and that represents
a group of sequence of information. If that sounds vague, that is because it is! It has applicability
to the ARC-A6GI challenge, but that is far from the only situation this might be applicable to. Analyizing
the results of a collection of science experiments comes to mind immediately as another application.

It also consists of, given the right selection, the information needed to completely rebuild a piece
of evidence used to construct the collection. For example a grid of integers, or a string of text. A
latent collection in this particular experiment will be a tensor that has dimensions associated

## What is this specific latent collection?

The latent collection representation that is tied to this project will expect to 
be built from batches of collections of sequences, and will try to maintain that structure.
That is, it will expect to be built from data:

* You have a list of discrete pieces of evidence, List[latent_evidence], which are in a sequence
* You structure the sequences into a generally useful collection, List[List[evidence]]
* You gather these into batches, List[List[List[evidence]]]

Now, what is that latent evidence? Basically, an encoding of a single piece of evidence
such as a string of text or, more relevantly, a grid of ints. Importantly, the latent_evidence
is reversable back into it's original representation.

## What is the technical contract of a latent collection

At the moment, a latent collect is expected to consist of a mask and data tensor.

**mask: A mask tensor.**

* dtype: float, but 0 or 1 (1 active)
* Purpose: Since we are dealing with sequence data and batching,
  some sequences may end early. Additionally, when batching collections
  shorter collections may end early as well. This accomodates that
* Design: Empty or padding elements filled with zero. Nonempty filled with one.
* Shape: (batch x collection x sequence)

**data: the data encoding**

* dtype: float
* Purpose: Encodes the actual data. May contain padding
* Shape: (batch x collection x sequence x  num_embeddings x embedding_dim)

It will be the case that batch addresses the batch, collection the collection element, sequence
the position in the sequence, and num_embeddings plus embedding_dim are specific to how the latent
evidence encoding was made

## What is the point of a latent collection?

Basically, a latent space collection allows us to both model the environment and make predictions. We can
do this by means of transforming latent collections, and viewing these latent collections. We can also recieve
feedback by comparing latent collections



