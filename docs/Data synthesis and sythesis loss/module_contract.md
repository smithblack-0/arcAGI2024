# Data synthesis

The provided training data will be completely inadequate for what is planned. As such, data
synthesis is going to be part of the game. Furthermore, this synthesis engine will, itself, need
to be trainable to make the tutoring mechanism work.

## Considerations

We need to support

* Tutoring: The model better be differentiable all the way though
* Unit synthesis: We should be able to synthesis for one unit
* Integration synthesis: We should be able to later synthesis for the entire group

It will thus likely end up being the case

* It is better to perform synthesis in the latent representation if possible.

## Cases

Generally, there are a few phases of synthesis to look at:

* type_specific_synthesis: We need a way to synthesis a piece of evidence.
  * probably based on an input vector which starts as random noise.
* transform_synthesis:
  * We need to be able to synthesize various transforms we wish to apply to the collection
* collection_synthesis:
  * Need to synthesis items in an entire collection
  * Should use type_specific_synthesis to work.

## Tutoring Process

Synthesis will generally be supported by a tutor. The tutor's purpose in life is
to ensure the examples that are being generated are within a difficulty range that makes
it easy for the model to learn. 

**Tutoring with distributions**

Tutor's will generally operate by predicting the next item based on some distribution, which it will
then encode and targ



