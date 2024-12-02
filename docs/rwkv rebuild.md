# Outline

Rebuild rwkv as a completely recurrent reversable cell. Towards this end:

## What is a reversable architecture

A reversable architecture, for the purposes
of this paper, is an architecture in which it is 
the case that a forward pass can occur, and then 
gradients are propogated backwards with a specialized 
reversable backwards pass that takes constant memory
and rebuilds the activations from the prior states
and the inputs.


## Reversable token mixing

The primary challenge to making a reversable recurrent
cell out of the RWKV model is reversing token mixing in 
constant memory. 

This term is defined in terms of something like:

$f(x_t, u_r, x_{t-1}) = y = (µ_r ⊙ x_t + (1 − µ_r) ⊙ x_{t−1})$

While x_t can be recalculated from the inputs, in order
to build a reversable computation graph we need x_{t-1}.
A reverse of this operation would be:

$g(x_t, u_r, y) = x_{t-1} = (u_r*x_t + (1-u_r))/(1 - u_r)

Now, this in and of itself is not numerically stable. However, 
it turns out f(g(x)) IS, except at zero. So we phrase this 
as something that runs using

$f(x_t, u_r, x_{t-1}) = y = (µ_r ⊙ x_t + (1 − µ_r) ⊙ x_{t−1}), abs(1-u_r) > eps$
$f(x_t, u_r, x_{t-1}) = y = (µ_r ⊙ x_t, else)

Which will now be stable. The only catch here is we need
another bit of state - namely, we need y, the token mixing output state,
to do the reversing.

## Rebuilding timestep mixture

Timestep mixture is rebuilt to use the above modified mixing strategy. It also
is modified to operate in the recurrent format:

$ s' = diag(w)*s + k^T*v $

But returns the additonal bit of state

$y = lerp(y)$

Reversing then becomes a simple matter.

## Rebuilding channel mixture

Channel mixture just needs to be rebuilt
to return additional gate channel state.
The channel state also now returns a recurrent
state regularly.

## Is fully reversable?

Yes, it can be reconfigured to be fully reversable, and
with minimal caveots. We simply need to implement 
a reverse step, a reverse version of the model, and 
modify how the mixing is happening to support the new
mixture function. 

And after that, we can drop in pretty much any parameter
set up to eagle.

After eagle, we would need significantly more 
hidden state to be passed along to make this reversible.
In other words, I can hack the model to use the existing
parameters and just change the way it is being used.

It would probably be better though to start with rwkv 4.
Then get more complex from there. 


Which is fantastic. It will save me the hassle of pretraining it.
It will of course now require a specialized loss function though. 

