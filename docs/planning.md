# Cognition Planning c Model:

## Introduction

What do I want the model to be able to do. Basically, I would like to be able to:

* Allow the exploration of various chains of actions and examine the resulting effect on the environment, to try to convert an environment from an intial state to a final state. 
* Incorporate multiple different examples by means of a related submodule.
* Allow backtracking and differing computation times
* Be modular so that other problems are still tractable.

To do this, we:

* Extensively lean on Latent Representations 
* Use Adaptive Computation Time to allow for multiple rounds of processing until the task is done
* Explictly include a "testing" phase inwhich the model tries a proposed plan to see if there is any sort of gain.

## Math

Assume for the moment that there is a set of input-output pairs that are present in a list as examples. That is we have a list of content $x_i = (input_i, target_i)$. We also have associated with this a function $f$ which accepts an input case and a plan $\omega_n$. It attempts to execute the plan in sequence.

$\hat y_i = f(x_i, \vec \omega)$

Our task is to figure out what parameters and length of plan, $\vec \omega$, gives something with the lowest loss when compared to the target, over the majority of input-output pairs. It is hoped this will then generalize to an unseen case x.

Basically, we are figuring out the rules of the transform.

## Dependencies

We are going to assume we can encode input, and output, into a latent representation. This will almost certaintly be implemented as a variational autoencoder of some sort. We will assume we can encode into, and back out of, the representation. We will be able to train this using generated data

We also will go ahead and assume we can create a model that accepts a list of embeddings representing transforms, along with an initial LR, and applies the transforms to the LR to get a final state. This, again, can be generatively trained.

## Process

A general idea of what we are going to do is as follows. We start by encoding the environment into it's latent format and setup all relevant state vectors. Then we propose actions

* Process Environment: None, Examples -> LatentInput, LatentState, LatentOutput
* Start state: -> State, Control, backtrack_loss
* ACT
    * Produce Latent Thoughts: LatentInput, LatentState, LatentTarget, State, Control -> Control, Thoughts, State
    * Propose Plan: Control, Thoughts, PlanDistribution -> PlanDistribution, (Actions, ActionWeight), += backtrack_loss
    * Process Environment Response: (Actions, ActionWeight), Examples -> LatentInput, LatentState, LatentOutput, Feedback
    * Merge feedback: Control, feedback -> Control
    * Act controller: Control, cumulative_probabilities, cumulative_plan, plan -> is_halted, cumulative_probability, cumulative_plan
    * if is_halted:
        break
 * Intrinsic loss
     * sum of backtrack_losses. weak
     * sum of plan probabilities. Stronger. More active elements -> harder plan to execute.

The output will now be a plan and a dictionary of losses.
      
        
Lets look at some of these in more detail
  
### Latent Thoughts:
  
This is basically the controller for the model. It is responsible for considering what has happened before and producing responses that are believed to bring us closer to a solution. It will look at the example input, example output, and current state. It will then generate something it thinks might help.

**What is it**

This is basically a generative transformer for a "Chain of thought" that a model may have, internal content that is used to proceed towards a solution. It does not directly contribute to the solution, but rather contributes little pieces that can be assembled downstream into an entire answer.

Each thought IS responsible for controlling a chain of thought that might lead to the answer. Additonally, they MAY be turned into actions that can be utilized as part of a plan.

**Contract**

You should accept:

* LatentInput: An input in the relevant Latent Representation format
* LatentState: An environment state in the relevant Latent Representation format
* LatentTarget: A targetted output in the relevant Latent Representation
* State: Whatever recurrent state information we will end up using.
* Feedback: A feedback embedding.

You should return:

* Control: An embedding: (B x E). Used to provide controller communication information
* Thoughts: A vector of embeddings. (B x N x E) This will be used for attention, and each time we create a new control element we should include it in thoughts automatically.
* State: The internal state used to track information. Probably the same as thoughts, honestly

**Implementation**

In practice, this will likely be implemented as a generational transformer that will just provide an updated thoughts list each time. That seems simplest. We also return the generated element as the control feature.

This will have to handle working with the latent input, latent output, and such of course. 

### Planner Model:

This segment uses an attention-like mechanism to create a series of "actions" with action probabilities associated with them, which are expected to be executed in the sequence given. 


**What is it**

When an action is executed, the resulting LR will only be updated in proportion to the action probability - which means an action can be ignored by setting the probability equal to zero. This allows the model to backtrack if something is not working.

**Contract**

You should accept: 

* Control: The attention query. (B x 1 x E), basically. Will tells us what to focus on
* Thoughts: The latent thoughts. (B x N x E)
* Prior Distribution: The prior MDN distribution.

You should spit out:

* Plan Distribution: A MDN distribution representing actions and their probabilities
* Plan (Actions, ActionProbs):
    * Actions: Built out of the thoughts directly, basically. Sampled. (B x N x L)
    * ActionProbs: The probability of an action occurring. Sampled. Each item in [0,1] by sigmoid activation. (B x N x 1). 
* Divergence Loss: A loss based on the difference between the last and current distribution.

**Implementation**

Basically, what we are going to do is perform attention without multiplying the values by the key-query matrix.

The control element will serve as the query, the thoughts element will serve as the keys. Actions will be produced by the value. We compute the results, then return the whole schebang.

We will generate an MDN distribution however, which when sampled gives one of several output embeddings. The first element of the output tensor will be the probability, and everything after it is an embedding - making sampling corrolated. We will then sample from the distribution, with a temperature controlled by Control. This will actually be a sort of beam search.

We will ALSO compute a KL divergence between the current and prior distribution. A significant change will be considered indicative of backtracking and will be penalized. 

## Environmental Interface

The environmental Encoding/Decoding mechanism finds it's purpose in testing plans and focusing the environment on whereever things are performing worst.

**What is it?**

This is complicated to explain. Basically, it is going to be an interface of sort to more primitive models or even the environment itself. The encoding step will consist of taking an environment and plan then popping out the most relevant input state, plan execution, and output state. 

It ALSO must contain a decoder capable of taking a plan and trying it out

**Contract**

You should accept:

* Examples: Input-Output example mappings. Presumably in raw format.
* Plan (Actions, ActionsProb):
    * Actions: Embeddings (B x N x L), compatible with the decode mechanism
    * ActionProbs: Probability tensor (B x N x 1) compatible with the decode mechanism
    
You should return:

* LatentInput: A latent reprentation showing the input example
* LatentState: A latent representation showing the state after applying the plan
* LatentOutput: A latent representation showing the output example 
* Feedback: Some feedback on how the decoding went. (B x E)

**Implementation**

There is a LOT to implement here. Lets talk models first

* First, encoding and decoding into and from latent space will probably be dealt with by a dedicated variational autoencoder work 
* Applying a plan can be performed by means of a feedforward network that accepts the current LR and the plan action, predicts the result, and then updates only if the action probability is high. A LSTM controller would be feasable as well. This would repeat until all portions of the plan have been processed.

Now, how the implementation works is:

* Encode all inputs and outputs
* Process plans on inputs in parallel
* Rate plans and generate feedback based on states and targets.
* Combine inputs, outputs, state, and feedback using rating weights as in attention. This will focus us where we are having issues, hopefully.

## Merge feedback

This is very straightforward. A feedforward or LSTM merges the feedback and control elements. The output is the same width as the control input.

## Act Controller

The act controller is exactly what it says on the tin. It is responsible for updating the cumulative plan as the controller works and the cumulative probabilities

**Contract**

You should accept:

* Control: (B x E), The control embedding for the iteration.
* cumulative_halting_probs: (B x N), the cumulative halting probs
* cumulative_plan: The cumulative plan, in whatever pytree it might be. Consists of tensors of (B x N ...) format in some nature.
* residuals: the residuals. (B x N)

You should return:

* is_halted: Whether everything has reached a halted state
* cumulative_halting_probs: The halting probabilities
* cumulative_plan: The cumulative plan
* residuals: The residuals.

**Implementation**

Currently, we expect to project control to create halting probabilities. Then we would use this to perform adaptive computation time on the fly, basically. 

