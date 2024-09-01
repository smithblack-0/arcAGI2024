# Overview

## Models

* The architecture being described is a multimodal model using a common context for generation purposes.
* Modal content is generated in "blocks" of fixed length and defined shape. 
* A common transformer decoder is used to process content and produce results in each block. Each
  block has it's own vocabulary associated with it to produce predictions.
* All training data is processed using "next sequence prediction" with positional encoding injection.
  The positional injection only requires the position since the start of the block, making it easy
  to inject whether predicting or sampling. All training data has a shape, which could be something simple
  like the length of a text, or more complex like an image shape.
* The produced embeddings are made into predictions, and loss, using a mode-specific vocabulary.
  This is true whether training or evaluating. This mode-specific class can also do distribution
  sampling for independent generative processing.
* The correct mode to generate in, and the shape of the mode, are ALSO predicted/sampled
  features. After each block is finished generating, the model is forced to predict which of the
  N supported modes to generate next. After this, it is made to predict the dimensions of the block 
  within the mode/sampled. 
* To keep things sane, the mode-specific Embed mechanism, mode-specific Target mechanism,
  mode_specific Loss mechanism, mode-specific Distribution mechanism, and mode-specific Sample mechanism are all located in one mode
  support class. These respectively should convert raw data into embeddings and shape targets,
  convert the embeddings and shapes into shifted embeddings (with start token) and attention masks,
  convert output embeddings into a predictive distribution, convert a distribution, targets, mask combo into 
  a loss, and sample from a provided distribution.
* Asyncronous training is supported to offset the separate vocabulary cost. Training data is 
  expected to be formatted in terms of a list of blocks, each of which specify a mode, a shape,
  and the actual data content. When a block needs to be evaluated, it is dispatched to a queue that
  accumulates the context, target pairs until it can put together a batch out of the training examples.
  Once a full batch is ready, it is run, then training resumes moving from that point forward
* Asyncronous evaluation is also supported. Again, a process of block request, generation, dispatch
  is utilized.
* A time to live mechanism ensures that if a case has not been processed by a certain point,
  it can be forced to run as an incomplete batch.
* A central dispatcher reads in training or test examples, dispatches them to mode-specific
  caches, and recieves the resulting embeddings back.
* 
## Models

Basically, the plan is to hijack any existing training or evaluation pipelines.

* Each model will be viewed in terms of its traditional pipeline, with three extra complications
  * The first complication is that it will be provided with a context tensor, which it should 
    presumably use to condition generation. For some problems, like transformer decoders, this
    is not unusual.
  * The second is that it will be generating to a shape, and needs to produce and provide feedback
    based on this shape
  * The third is we need to define the shape as a thing based on the training data, and this ALSO
    will need to be predicted. 

### Text

* Tokenize