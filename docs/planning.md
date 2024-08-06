# Overview

This is a planning document for the ARC-AGI intelligence project

The plan is, in a general manner, to:

* Create a considerably more sophisticated architecture designed to investigate and plan
* Pretrain the pieces
* Put them together and fine tune on the application.


# Primary Input Processing

Input data for the model will consist of collections of sequential examples, with the model task
being to find a rule to run a transform to transfer between the sequence examples. 

### Dependencies

* cases Dict[str, nn.Module]: A mapping of the type of data to the latent encoder to use to process it.
  * Different data types need different latent encoders.
  * You will get out a tensor of shape (N x embeddings). N is fixed
* shape predict

### Intake Formatting Contract

Raw input data will consist of

* Individual data entries consisting of a dictionary of {type: str, payload: Any}

These would then be collected into cases:

* Case: List[Data entries]

Which are then more broadly collected into a collection

* Collection: List[List[DataEntries]]

### Output Formatting Contract

The output will consist of a collection of gram embeddings, with associated masks. 
They will have a shape associated with them, along with a sequence mask, that looks
like:

* collection_embeddings: shape (batch x example x sequence x gram_layers x embeddings)
* collection_mask: shape (batch x example x sequence)

Where:

* batch: The batch number we are working on
* example: The collection number of the example
* sequence: The sequence location. 
* gram_layers: The number of gram layers that were encoded
* embeddings: The embedding dimensions.

The mask is used to handle cases in which the sequence lengths are different between
the examples. 

### Des



Data input can and will consist of lists of lists containing sequential input features. In particular
it may consist of a sequence of lists which each contain an example, and which may then in turn contain bits of 
sequential data. 

This will be processed into a grid of gram encodings, which can then have their positional information seeded
within them.

This format is designed to be highly modular and general. 


## Project Parts

Input processing:
* Latent represetations: How to convert a grid of ints to a latent representation and get it bck
* Size encoding: Represent size-based features like num tokens or grid size in a clean manner.
* Sequence Processing: How to represent a input-output latent representation as a feature we can learn from
* Collection Processing: How to represent an entire example collection in a manner we can learn from.

Gridsize Prediction:
* Predict output grid shapes given input-output grid size pairs.

Transformations:
* How to represent a transformation from one latent representation to another
* How to encode this transformation in a way that is differentiable.

Data management and generation:
* Data mocking 1:
  * Mock input grids.
  * Tutoring with mock input grids
* Data mocking 2:
  * Mock input sizes
  * Tutoring with mock input sizes
* Data mocking 3:
  * Mock transforms with exampels
  * Tutoring with mock examples
* Data processing: 
  * Extending the input examples.

* Mock int grid generation
  * How to generate mock input grids
  * How to generate mock input grids in a way that preferentially challenges places having issues
* Mock transform generation
  * How to generate a sequence of transforms 
  * How to generate a sequence
* Mock data generation: Inputs, actions, and markov models.
* Data extension: Getting more out of less for finetuning.

Cognition Unit:

* General Cognition: How to create a general cognition structure 
* Hypothesis exploration: How to cause the cognition unit to explore different cases to rapidly gain informaiton
* Encoding commands: How to encode actions or sequences of actions from the cognition unit in a model agnonistic way.

Tutor Units:

* How to make a tutor unit.
* How to ensure it remains differentiable.

# Main pieces
