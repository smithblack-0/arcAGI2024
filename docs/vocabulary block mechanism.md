# Multimodal Vocabulary Mechanism

This module implements the multimodal vocabulary mechanism, where data is represented as "mode-dimension channels." This structure accommodates the multimodal nature of the architecture, allowing data to be processed in different modes (e.g., text, image) while maintaining compatibility with batch operations.

## Tokenized Data Flow: Mode-Dimension Channels

In this system, rather than treating data as a simple sequence of tokens (e.g., text as a chain of integers like [1245, 3563]), data is passed in terms of **mode-dimension channels**. Each mode of operation (e.g., text, image) is associated with a mode integer, and the actual data is passed as a channel of integers that encode content in that mode.

For example, text data in mode 2 might be represented as:
`[[2, 1245], [2, 3563]]`

However, some data, like images, may require more than one dimension. To handle this, data is passed using a 1 + D channel format, where D is the number of maximum dimensions for the mode. If any extra dimensions aren't needed (e.g., for text), they are filled with zero-padding. For example:
`[[2, 1245, 0], [2, 3563, 0]]`

This multimodal format ensures the architecture can process diverse data types simultaneously without losing the structure specific to each mode.

## Schemas

The shape of data for each mode is defined using a **schema tensor**, a static quantity that informs how the model layers are set up. The schema tensor is a 2D integer tensor, where the first dimension indexes the mode and the second dimension describes how many elements exist for that mode. Unused dimensions are padded with zeros.

For example, for text data with a vocabulary of 10,000 words and images with a size of up to 512x512, the schema might look like:
`[[10000, 0], [512, 512]]`

The schema is immutable once created and is critical for both setup and runtime operations. It ensures that data flows correctly through the model and is properly embedded for training.
