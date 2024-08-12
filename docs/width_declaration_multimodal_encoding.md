# Length Declaration Multimodal Encoding Decoding


## What is it?

Width declaration multimodal encoding/decoding is a way of representing data in differing
modes while operating in an encoding or decoding tranformer-like context such that a single
model can handle multimodal sequence-based data.

## How does it work?

Data is encoded, or decoded, in large contiguous chunks called "Mode Blocks". These blocks
have shape (... x block_length x embedding), and positional context which can be injected.

The first three embeddings of a block are called the Header, and the remaining section of the block
is the Payload. The "declared length" of the block is related to the encoded data, and the actual
block length is equal to declared_length + 2.

The first Header is the Mode Embedding. It specifies the type of mode of the block. The second
Header embedding specifies the Shape of the block, and will be associated with a relevant ND
integer. The declared length is the product of this Shape. For instance, with a 10x10 image, declared
length would be 100. The third and final embedding is the Sequence embedding. It is related to how 
many Mode Blocks have been created or seen before. This concludes discussion on the Headers.

The payload gets context injected into it, and this context includes block mode type, positional
encodings, and sequence. The positional encodings are based off the declared Shape, which means 
that multidimensional encodings can be supported by flattening the encodings for the representation.
This information can be injected when encoding OR decoding, allowing easy tracking of, for example, 
individual pixel locations.

Predicting the correct Mode and Size then becomes one of the features that is trained when
decoding, allowing operation by prediction in training or evaluation mode.

## Encoding

To encode, start with a sequence of multimodal information. It should be structure data with
something that can be considered a "shape". Turn it into a sequence of information that 
consists of mode, shape, payload, and sequence position.

* mode: 
  * Shape: (... )
  * Type: Integer
  * Purpose: Specifies the mode we are generating in as a vocabulary entry.

* shape:
  * Shape: (... x K), K is number of supported dimensions
  * Type: Integer
  * Purpose: Specifies the shape. The declared shape will be the product across K.

* sequence_position:
  * Shape: (...)
  * Type: Integer
  * Purpose: Specifies what position in the multimodal sequence this term was.

* payload:
  * Shape: (... x items x embedding)
  * Type: Float
  * Purpose: Is the actual payload we are attempting to encode.
  * Detail: It should be the case that items is of length product of shape.

  
### headers

Making the headers is quite straightforward. 

* Grab the correct term from the mode Embedding layer. This is your mode embedding
* Run the shape through the mode-specific Shape Embedding layer. You now have a shape embedding
* Run the sequence term through the Sequence embedding layer. Probably a feedforward network for now

Once this is done, we have the three required headers.

### context injection

The context is injected along the entirety of the payload. It consists of the sum of the headers
The positional context is injected along the entire defined payload. It consists of a Block Context
which is universal across all elements of the payload and a positional encoding

The block context is very straightforward to create. It is the sum of all three embeddings. This
is injected on every payload element in the block. The positional encodings, meanwhile, are created
based on the distance since the start of the block and based on the block shape. The shape is provided
to the positional encoding mechanism - furthermore, the positional encoding mechanism is expected to use
such information to provide ND encodings that can be used to distinguish between different dimensions.

The positional encodings are sophisticated and both count up from sequence start, along with
count down to sequence end. 

### Process completion.

The entire encoding process can now be viewed. We take the multimodal data, format it correctly,
and create headers, payload, context for each. We inject the context into each payload,
and can concatenate the entire shebang together. We now have a WDM encoding.

## Decoding: Training

Decoding proceeds in a somewhat similar manner, and still manages to inject positional information
in the correct locations while training the model. We can use teacher-forcing and next embedding prediction
like in a standard generative transformer decoder. 

### Data

Like in the encoding case, we take our targets and create the four key tensors out of them:

* mode: 
  * Shape: (... )
  * Type: Integer
  * Purpose: Specifies the mode we are generating in as a vocabulary entry.

* shape:
  * Shape: (... x K), K is number of supported dimensions
  * Type: Integer
  * Purpose: Specifies the shape. The declared shape will be the product across K.

* sequence_position:
  * Shape: (...)
  * Type: Integer
  * Purpose: Specifies what position in the multimodal sequence this term was.

However, a notable difference exists in the payload. While the encoded payload itself will
still be embeddings, the targets used with the predicted distribution may vary depending on content.
For instance, although text would generally consist of integers, image data might be a set of 
3 numbers for RGB.

* targets:
  * Shape (... x items x ?)
  * Type: Varies
  * Purpose: Original data

* payload:
  * Shape: (... x items x embedding)
  * Type: Float
  * Purpose: Is the data after encoding.
  * Detail: It should be the case that items is of length product of shape.

Also novel is the Padding tensor. Decoding requires specifying ahead of time how
much you have to say, and sometimes you might not need all the specified room. In that
case, the padding tensor can be used by the model to specify these elements do not matter.
A random amount of padding can be added during training to get the model used to this
process

* padding:
  * Shape: (... x items)
  * Type: Float, between 0 and 1.
  * Purpose: Tells us whether this is data (1) or just padding (0.)

We then do, in fact, encode this information as described above. These will be
used as part of the teacher-forcing sequence.

### Block Generation

Decoding operates in a cycle in which blocks are generated, processed, and predicted.

The cycle begins by generating the three headers. This process will lock the model into
a Mode and Shape over the remaining generative cycle. A Mode Distribution, and a mode-specific
Shape distribution, will be used to predict these features - and during training, the correct
mode decision and shape decision will be teacher-forced to configure the model correctly for
the subsequent embeddings. 

Once all three headers have been generated, we proceed to engage the mode-specific distribution
and loss mechanism for the active mode. We generate our embeddings, gather losses between targets
and predictions, and accept the teacher-forced actual token with context injected into it. This
will include positional context, letting us track the position of information. We also predict
from each created embedding whether it is padding or not. This, again, is included in the
computation of loss.

Once this is over, we immediately start to make a new block, including the three new headers, and
repeat until over.

Some final notes.  One of the modes is reserved as a END of generation signal to tell us when
to stop making tokens. This should be predicted by the model. 

## Decoding: Eval

When operating in evaluation mode, we do away with the targets. Instead, we use the
trained distributions, and treat selecting the mode and the shape as sampling activities. 

We sample from the mode, then use the mode-specific shape decoder to sample the shape, then
we create the sequence embedding. Then, we go and generated, predict, and sample payload entries 
one at a time producing the output information. We repeat this for all modes as relevant.
