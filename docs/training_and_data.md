# Start

* batches of sequences of collections.
* Prediction mechanism exists.
* No one collection has all the information needed for a solution

## Model contract

We split the training data up into two portions consisting of the examples, and the test cases.
Furthermore, each of both is split up into inputs and targets. We then attempt to learn a tensor of
"rules" that is shared between the sets and that lets us create the targets out of the inputs.


## Training tasks

Training tasks are needed for pretraining purposes. They must contain the following properties

* task