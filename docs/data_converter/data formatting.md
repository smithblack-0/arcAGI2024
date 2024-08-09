# Introduction

This concerns the formatting contract that raw data will have to be in

## Contract

The raw data zone contains information in its raw, tensorless format. It is possible to 
encode out of and back into this format from a latent collection

Raw data consists of "evidence", individual terms, "sequences",
which are a ordered list of elements, and then "collections" which are groups of sequences. These 
groups can then be put into batches. 

* evidence Tuple[str, Any]: A data type and a data payload
* sequence List[evidence]: A sequence of data to understand
* collection List[sequence]: A list of sequences. In our case an example collection

These can finally be put together into

*batch: List[collection]:

Converting to and from this is done by means of type_specific converters that are applied on the raw
data, and which then produce a latent collection tensor