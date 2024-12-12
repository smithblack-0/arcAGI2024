# Rethinking Generative Reversible Timestep Recurrent Networks for the Transformer Era: Training over infinite contexts, and the optimization objectives to exploit it.

* Small blurb to talk about modern reversable recurrent models.
* We propose a strategy to design high capacity recurrent models
  for the transformer era.
* Extra emphasis is placed on designing an architecture that is highly modular 
  and easy to implement. New insights can be inserted easily.
* We also briefly highlight how one modern recurrent model, RWKV, 
  can easily be rewritten to satisfy this strategy.

# Prior work

## History of the transformer

* Attention, feedforward step
  * Parallize much easier. 
  * Better results
  * Quadratic complexity in time
* Beating the quadratic complexity
  * Linear transformer.
  * Others

## Work division of the traditional transformer cell

The traditional transformer can be viewed in terms of "spatial" and
temporal operations that view the past in order to make predictions. They
are composed of layers that perform:

* "Temporal" mixing
  * Produce effects based on what has happened in the past
  * Attention is the "traditional" mechanism
  * "Linear attention" is a rewrite of this that has the interesting property of 
     being rewrittable as a recurrent operation.
* "Channel" Mixing
  * Produces effects based on what has happened spacially
  * "Feedforward" is the traditional implementation, although others such 
    as spacial mixing (RWKV) have also been attempted

## Refactoring as a recurrent process

Recurrent transformers are an active domain of research. They are recurrent
representations of linear transformer processes that have shown some promise. 
Under these interpretations, any "Temporal" mixing becomes a recurrent
update step.

(Do literature search)

## Optimization and training of Generative Natural Language Architectures

* Generative Language architectures have been trained on a task known primarly
  on next token prediction in a pretraining event
* Usually trained on "time-parallel" model types such as the transformer. 
* Usually trained on batch widths between 1-30.

## Reversable Timestep Cells with Feedback

For the purposes of this paper a reversable timestep
cell is said to incorporate "feedback" if it is the case
that new recurrent states are not constructed from the old
using the "no-feedbck" condition. 

### Coupling feedback

One very popular reversable cell consists of:

$s_1' = s_1 + f_1(s_2, x)$
$s_2' = s_2 + f_1(s_1', x)$

Where $s_1'$, $s_1$ are recurrent state 1, and $s_2'$, $s_2$ are recurrent state 2, 
and $f_1$, $f_2$ are some sort of gates. This gate is completely numerically stable under reversal,
as any error that occurred in the forward step will be compensated by the same error occuring in the 
backwards pass. However, it has capacity issues.

### Reversable LSTM or GRU

One way around this is to try to simulate a LSTM or GRU that is timestep reversable.
But that also has issues. It tends to fail at the job if no erase gate is provided,
while if one is provided irregularities between the forward and reverse step torpedo
the process.

(MacKay et al., 2018)

# Rethinking the Recurrent Transformer Cell for Reversibility over Arbitrarily Long Token Length.

## Recurrent Transformer Flavors



## The Feedback-Control Paradigm

* A way of writing or rewriting recurrent transformer cells to make them capable of training
  over infinite context lengths. 
* Most recurrent transformer cells already mostly satisfy this paradyn, allowing minimal modifications
* 

### What is it

The feedback-control paradym is henceforce established
as the situation in which

* There is one layer performing feedback on inputs over multiple timesteps. This then goes into
