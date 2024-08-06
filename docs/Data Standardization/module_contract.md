# Introduction

The data standardization process has the purpose of transforming a machine-learning hostile
enviroment into something in a standardized format which is much easier to understand and even 
predict. 

It consists of defining the mechanisms that lets a collection of sequential examples in various modalities
be converted into a latent representation consumable by a model, or conversely converted from a latent
representation back into the original format. 

Each specific piece of evidence is reduced to a latent representation, but the evidence is not
then further reduces. 

Note that this is designed with accurate reconstruction and transformation in mind.

## Challenge

The elements of the world do not come in a fixed size. For instance, a string of text may be 
one word long or thousands, or an image may be 32x32 or 3200x3200. In essence, the world does not come in
one fixed size.

Additionally, information in the world does not come in one type either. There might be text data, image data,
int grid data, or other data types as well.

## Terms:

* latent_representation: a fixed width representation of data elements
* encode: Take from raw data into a latent representation
* decode: Go from a latent representation back to data.

## Pieces

* Primary Data Converter: Responsible for converting a batch of raw data into a latent representation, or backwards
* Type Specific Converter: Responsible for converting a specific type of data, like an image or text, with minimal loss.

## Data contracts:

Lets talk about the data contracts, both in terms of the raw and encoded data. Basically, 
the raw data from the input side will consist of tri-nested lists of data features, then sequences,
then example collections, then batches.

**Raw data**

Lets talk about what data needs to look like before going to be encoded, or what will
pop out when getting decoded. Raw data consists of "elements", individual terms, "sequences",
which are a ordered list of elements, and then "collections" which are groups of sequences. These 
groups can then be put into batches. 

* evidence Tuple[str, Any]: A data type and a data payload
* sequence List[evidence]: A sequence of data to understand
* collection List[sequence]: A list of sequences. In our case an example collection

These can finally be put together into

*batch: List[collection]: 

The batch as shown above is what would be encoded

**Encoded data**

Once encoded, we basically promise every element will be mapped to a particular shape by
latent representation, and we place everything else in a single tensor

In particular, we learn something such that for any element we can map it to a latent representation
encoding of fixed width:

* element -> latent_representation
* latent_representation: Shape (N x embedding)

Where N is the same across all type_specific_converters. We then specify that the encoded format 
of the above encodes the data along the other dimensions and provide a mask where the sequence was
inactive. 

* batched_collection_encodings: Shape (batch x collection x sequence x N x embedding )
* batched_collection_mask: Shape (batch x collection x sequence)

## Design

Basically, we are going to split this job into two modular pieces. There will be the primary converter
and then the task specific converters. 

* primary_data_converter: Responsible for converting into or out of batched_collections_encoding
* task_specific_converters: Responsible for converting a particular data type. Highly modular.

**primary_data_converter**

This is responsible basically:
    * For maintaining a list of type_specific_converters
    * For encoding and decoding everything that does not have to do with encoding a piece of evidence

**type_specific_converters**

These are responsible for converting a particular type of evidence, like a string or an image.

* They must be capable of embedding presampled items, or a distribution. For instance, a list of tokens or 
  token probabilities.
## Concluding thoughts

Once everything is in terms of this collection encoding, a variety of tasks can be performed against it. This
can now be further processed into insights, or attempted to be transformed by, for example, next sequence
prediction.
