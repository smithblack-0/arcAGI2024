Decent start, but I think I would like to restructure this. Instead of the current approach, we are going
to talk about how the core of the recurrent transformer architecture is typically a reversable architecture,
then broaden that talk to discuss how to handle some edge cases.

## Recurrent Transformers are Extremely Compatible with Token Reversable Timestep Logic.

**Key ideas**

We

* Identify that the standard recurrent transformer formulation and it's derivatives are easy to make timestep reversable and quite numerically stable
* Identify that additional models incorporating feedback mechanisms tend to do so in a manner that contains the complexity of the update in a singular layer interacting in a hidden state that is not reused in further computations. This is then fed into the transformer formulation, "amplifying" the feedback
* Observe that this means that by replacing the feedback mechanisms with a reversable analogue, such a model immediately becomes reversable. 
* This reversability means we can find the previous hidden state from the last timestep given the current hidden state and the token.
* APPENDIX: Discussion on reversable reformulations of several recurrent transformer architectures. 

** What is a Token Reversable Timestep  (TRT) model?**

* Basically, it is a model where you can find the recurrent state information from the last timestep given the current timestep
  and the token/embedding
* It means you have a model that recomputes the activations then uses them to reverse the model.
* The core condition that must be satisfied is that recurrent updates must satisfy the "Write-First" condition. This means
	* Any updates to recurrent state are performed BEFORE performing output read actions. So always x_out = f(s', x), never x_out = f(s, x)
	* Without this condition, and assuming x' = f(s, x), s' = f(s, x'), we observe that we need to execute s = f(s', x') which requires knowing 
          s to get x'. This is circular, and can never be solved.
* Stability (but not capacity) of training is guaranteed when using additive writes: s' = s + g(x), so you generally want that.
* Net effect is that so long as you can fit the graph to process one token in memory, you can cover an infinite length of context. 
* No free lunch though: Eventually, the magnitude of your updates will be small in respect to the state tensor, causing no actual update to occur

**The TRT Compatibility of Core Recurrent Transformer Logic**

Most recurrent transformer processes have some flavor of step where they perform:

k, v, q etc = L(x_in)
s' = s + G(k, v, etc.),
x_out = K(s', q)

Which is inherently timestep reversable so long as x_in does not depend on s. That is, so long as x_in 
satisfies a "no-feedback" condition this is immediately reversable as long as you can compute 
x_in. Which is fantastic news - it means the majority of a recurrent transformer computation can be 
represented in a reversable manner. I consider myself very lucky that the industry standard seems to 
satisfy this condition, as it makes it very easy to rewrite this reversably as:


k, v, q etc = L(x_in)
s = s'- G(k, v, etc.),
x_out = K(s', q)

Now, so long as you can recompute everything that came before this, you can compute the 
current timestep.

**Tweaking Feedback layers for TRT: Containing feedback**


Some designs, such as RWKV, incorporate additional feedback from previous steps in a manner that may involve
mixing or other actions beside addition or subtraction. Such feedback may in fact be required for the effective
usage of recurrent models. This feedback is not naively reversable. But so long as we can "contain" the reversable complications
within the layer itself, we can make a drop in replacement that will be.

Consider the arbitrary recurrent cell written as:

s' = L(s, x_in)
x_out = G(s', x_in)

Where L may or may not have been reversable. Considering the internals of that layer, make
sure we now rewrite it such that:

s1, s2 = s
s1' = s1 + L1(s2, x_in)
s2' = s2 + L2(s1', x_in)
s' = (s1', s2')
x_out = G(s', x_in)

This fits in exactly the same footprint, which means we could get by just replacing the 
gate with a different time, without any other significant architecture changes. This is,
however, again predicated on the idea that x_in satisfies the no feedback condition with 
respect to s. It also means your existing architecture can be fine-tuned by keeping the 
existing parameters, except this region, when reconfiguring a trained model for reversable
action.

This is because such feedback is well contained to only have an effect on that immediate layer, a property to look for.


**Design principles for good candidate models**

Generally, any model that follows these principles will be an excellent canidate for rewriting as a reversable cell. Also,
following these principles when designing a model will make implementing reversability straightforward.

In order to be reversable from the next hidden state and the input, it is the case that ANY model satisfying this 
condition MUST satisfy the "Write-First" condition. You are simply not going to be able to invert the model without it

* Write-First: Any recurrent update should ALWAYS be executed before reading from the recurrent state to avoid noninvertible dependencies. 

Meanwhile, as best practices and concepts that make your model MUCH easier to reverse, you should generally be aware:

- Your primary transformer logic should perform an ADDITIVE update on recurrent state: s' = s + f(x'_in), where x'_in is independent of s. Most
  recurrent transformer formulations are, so long as x'_in is independent of the state of s. 
- You can optionally integrate additional feedback structures before the transformer of nature s2', x_in' = g(x_in, s2) that are designed to 
  be fed into the main logic, with the understanding that it will need to be replaced with a reversable structure. That piece modular
.

See the appendix for a discussion of how to make reversable some recurrent transformer architectures.

# Training a TRT model and training functions for infinite width context.
