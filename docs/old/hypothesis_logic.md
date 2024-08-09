## Premise

* By observing information about the things we are most uncertain of we can gain insight into how to 
  handle any task in the task space.

## Scenario

You have a set of inputs $I_i$, and their associated outputs $D_i$. You also have a mapping function
that accepts a set of weights $\hat D_i F(I_i,t)$. You seek the mapping function that will minimize
the reconstruction loss. Your examples are quite small, and must generalize to more cases

## Information gain process

Start by creating a transform to investigate.

* t_{n} = investigation_proposer(state, K_n)
* proposer's reward is based on maintaining high entropy for the hypothesis distribution. 

Use a gaussian mixture model or other model to create competing hypotheses regarding what will happen
when executing the transform.

* P(D_i| t_{n}, I_i) = create_hypotheses(t_{n}, K_n)
* reward will be based on how close the predictions turned out to be

We execute the transforms and get the actual information. We update the context. 

* $D'_{ni} = F(I_i, t_n)$
* $K_{n+1} = concat([K_n, (D'_n, I, t_n)])

We can now compute the rewards. 

* loss for transform proposal: \sum_i entropy(P(t_n))
* loss for create_hypothesis: -\sum_i log(P(D'_i|t_n, I_i))

The total loss for each layer will then be the loss over all iterations plus the objective loss.

* total_investigation_loss = -\sum_n entropy(P(t_n)) + objective_loss
* total_hypothesizer_loss = -\sum_n\sum_i log(P(D'_i|t_n, I_i)) + objective_loss
