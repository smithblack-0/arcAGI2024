# What is transform feedback?

Transform feedback is, roughly speaking, the process of getting information on
how well a job we did when using a particular transform. T


## Aspects of transform feedback

Lets look at the major portions of transform feedback we need to look into.

**Transform Accuracy**

Transform accuracy is whether or not we got the job done. Roughly speaking, if the 
targets and outputs look the same, we had high transform accuracy. Things which
need to be considered include:

* Did the transform actually result in the appropriate outcome?
* Up to how many terms of the sequence?
* What places are going particularly wrong

**Computational Expense**

A output of N terms can have required N+L auxiliary responses to decode. However,
in a perfect world, we would be able to associate each auxilary response with a single
output slot. This would be perfect efficiency. The difference, L, is the computational
efficiency.

We need to check how far away from this ideal, why, and use that information to reduce the 
computational efficiency to as small a term as possible. This may seem optional. It is 
not. An unrestrained model could easily add as many tensors as is needed to get it's specific
examples to work without this expense. However, with it as a loss factor, the model will be
forced to find compact sequences, which by occam's razor should usually be more general.

Factors to consider include:

* How do we tell the model about the computational efficiency?
* Do we let the model view it's auxilary sequences to get some ideas of what to optimize?