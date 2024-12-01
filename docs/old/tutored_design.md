# ARC-AGI Tutor, Solver, and Ranker Architecture with Reinforcement Learning

## Introduction

This document outlines an advanced architecture designed to tackle the ARC-AGI  
challenge. The architecture consists of three primary components: the Tutor,  
the Solver, and the Ranker. The Tutor is responsible for generating problem  
instances similar to the ARC-AGI dataset. The Solver attempts to solve these  
instances, and the Ranker predicts the likelihood that the Solver will  
successfully solve a given problem. Each component (Tutor, Solver, Ranker) is  
initialized independently, with no shared parameters or pretraining. The core  
model is a shared transformer decoder architecture, but each component  
undergoes independent training from the start. The Tutor is the only component  
that is influenced by a continuous constraint loss, and this constraint
loss is dynamically label smoothed. The constraint loss is fully trained, and 
then the label smoothing level is adiabatically changed to encourage cycles of exploring
possibilities then focusing on exploitation. Reinforcement learning is used to pick
out possibilities that have an outsized effect on the validation cases.

## Theory

Data is used to create a Teaching Distribution that is cooperatively  
refined and drawn from to get closer to a solution for the problem. This  
distribution serves as a proxy for the training data, enriched with controlled  
speculation and difficulty, and is refined through reinforcement learning  
based on its effectiveness in improving model performance on validation tasks.

### Data

* The foundation of the Teaching Distribution.
* Provides the initial context and examples for building the distribution.
* Ensures that the Teaching Distribution is rooted in real, relevant scenarios  
  while allowing for controlled exploration beyond the original data.
* We will also need a validation set of sorts as well.

### Teaching Distribution

* **Construction**:  
  - The Teaching Distribution is built from the training data but allows for  
    controlled expansion beyond it.
  - The original training data is embedded within the distribution, ensuring  
    that it remains relevant to the problem domain.

* **Speculation and Difficulty**:  
  - The distribution is trained to exhibit a specific degree of speculation,  
    allowing the model to explore new possibilities that the training data  
    alone might not reveal.
  - It is also trained to present problems of varying difficulty, targeting a  
    balance where about 50% of generated problems are solvable by the model.

* **Sampling and Refinement**:  
  - The Teaching Distribution is sampled to generate examples that the model  
    can learn from.
  - Reinforcement learning is applied to refine the distribution, focusing on  
    examples that lead to significant improvements in validation performance.
  - This refinement process continuously adjusts the distribution to optimize  
    the training experience, balancing exploration with the need for  
    generating useful, solvable problems.

### Model

* **Objective**:  
  - The ultimate goal of the process is to train a model capable of solving  
    the problem at hand, based on examples drawn from the Teaching Distribution.

* **Training Process**:  
  - The model is trained on data sampled from the Teaching Distribution, which  
    has been refined to present a variety of scenarios with appropriate levels  
    of difficulty and novelty.
  - Crucially, the model never sees the original training data directly.  
    Instead, it learns from the enriched and diversified examples provided by  
    the Teaching Distribution.

* **Isolation from Training Data**:  
  - The model’s exposure to the original training data is mediated entirely  
    through the Teaching Distribution.
  - This isolation ensures that the model generalizes from a broader range of  
    scenarios rather than simply memorizing the training data.

## 1. Overall Architecture

### 1.1. Core Model
- **Common Transformer Decoder**: The architecture is built around a shared  
  transformer decoder model that serves as the foundation for the Tutor,  
  Ranker, and Solver. While the core model architecture is common across  
  components, each one is independently initialized and trained, with no  
  shared parameters or pretraining.

### 1.2. Tutor
- **Objective**: The Tutor generates new problem instances that are relevant  
  to the ARC-AGI dataset but introduce enough variation to challenge the  
  Solver. It actually contains the teaching distribution.
- **Key Features**:
  - Generates examples with controlled difficulty, aiming for a 50% success  
    rate for the Solver.
  - Receives supervised learning loss based on how close the difficulty of  
    each example is to the 50% target, ensuring that the examples are neither  
    too easy nor too hard.
  - Uses reinforcement feedback based on the validation set performance to  
    refine its generation process.
  - The success probability is a schedulable parameter, adjusting the loss  
    without modifying the Tutor itself.
  - The Tutor is guided by the Ranker, which is part of the feedback loop. The  
    Ranker helps the Tutor adjust the difficulty and diversity of generated  
    examples, ensuring that the examples remain challenging yet solvable.
  - **Constraint Loss**: The Tutor is the only component that is trained with  
    a continuous constraint loss. This loss ensures that the generated  
    examples adhere to predefined constraints, guiding the Tutor to produce  
    examples that are within the desired difficulty range and are consistent  
    with the core assumptions of the ARC-AGI challenge.

### 1.3. Solver
- **Objective**: The Solver attempts to solve problem instances, both from the  
  original dataset and those generated by the Tutor.
- **Key Features**:
  - Transformer-based model (shared with the Tutor and Ranker) that takes  
    input-output pairs and a holdout input to predict the holdout output.
  - Trained using standard supervised learning techniques, possibly augmented  
    with additional loss terms to reward solving challenging examples.
  - **No Access to Training Data**: The Solver is completely insulated from  
    the actual training data. It never sees the original training data and  
    learns solely from the examples generated by the Tutor. This ensures that  
    the Solver’s learning is based entirely on the variations introduced by  
    the Tutor, promoting generalization.

### 1.4. Ranker
- **Objective**: The Ranker predicts the likelihood that the Solver will  
  successfully solve a given problem instance. It is important for controlling 
  problem difficulty.
- **Key Features**:
  - Binary classification model that estimates the success probability of the  
    Solver before attempting a solution.
  - Provides feedback to the Tutor to guide the generation of problem  
    instances with appropriate difficulty.
  - The Ranker is fully integrated into the feedback loop, receiving training  
    based on outcomes during both the generative and validation processes.  
    This ensures that the Ranker’s predictions remain accurate and that the  
    Tutor continues to generate relevant and challenging examples.

## 2. Training Process

### Continuous Constraint Loss

The training process is streamlined by using a continuous constraint loss,  
which is applied exclusively to the Tutor. This loss guides the Tutor’s  
generation process, ensuring that the examples it produces are both relevant  
and within the desired difficulty range. The continuous constraint loss  
replaces the need for a separate constraint phase, integrating constraint  
learning directly into the Tutor's ongoing development.

- **Dynamic Label Smoothing**: Label smoothing levels are adjusted dynamically  
  throughout training. This scheduling allows the Tutor to alternate between  
  more explorative phases (higher smoothing) and consolidation phases (lower  
  smoothing). Early on, higher smoothing encourages broader exploration, while  
  later stages focus on fine-tuning with lower smoothing levels.

- **Maintaining Core Assumptions**: The continuous constraint loss ensures that  
  the Tutor maintains core assumptions throughout its training, providing a  
  stable foundation as it evolves. This loss component guides the Tutor to  
  generate examples that adhere to expected constraints, ensuring consistency  
  and relevance.

- **Simplified Training**: By eliminating the need for distinct training  
  phases, this approach reduces complexity and makes the training process more  
  efficient. The Tutor continuously learns and adapts without needing to  
  transition between different modes of training.

## 3. Loss Function Discussions

### 3.1. Ranker Loss

The Ranker plays a critical role in guiding the Tutor by predicting the  
success probability of the Solver for each generated example. The Ranker's  
loss is primarily driven by how accurately it can predict whether the Solver  
will successfully solve a given problem instance.

- **Training on Generated and Real Examples**: The Ranker is trained using  
  both generated examples from the Tutor and real examples from any available  
  data. This dual training ensures that the Ranker can generalize well across  
  a broad range of problem instances.
  
- **Binary Cross-Entropy Loss**: The Ranker’s predictions are binary,  
  representing the likelihood that the Solver will solve a problem (success)  
  or not (failure). After the Solver attempts to solve an example, the outcome  
  (success or failure) is used as the target for the Ranker’s prediction. The  
  Ranker’s loss is calculated using binary cross-entropy between its predicted  
  success probability and the actual outcome.

- **Feedback Loop**: As part of the feedback loop, the Ranker’s loss directly  
  impacts the Tutor’s generation strategy. By accurately predicting which  
  examples are solvable, the Ranker helps guide the Tutor to generate more  
  useful and challenging examples, which in turn helps the Solver improve.

### 3.2. Solver Loss

The Solver's primary task is to generate the correct sequence for a given  
problem instance. The loss function for the Solver focuses on how accurately  
it can produce the target output sequence, especially when provided with new  
and challenging examples from the Tutor.

- **Sequence Generation and Teacher-Forcing**: The Solver is trained to  
  generate the correct sequence based on input-output pairs. During training,  
  teacher-forcing is used, where the correct output sequence is provided as  
  input for the next step in the sequence generation. This helps the Solver  
  learn more effectively by reducing the propagation of errors.

- **Logits and Cross-Entropy Loss**: The Solver generates logits at each step  
  of the sequence, representing the probability distribution over possible  
  outputs. The loss is calculated using cross-entropy between the predicted  
  logits and the actual target sequence. This loss directly measures how close  
  the Solver’s generated sequence is to the correct sequence.

- **Problem Solving Criterion**: A problem is considered solved if the  
  probabilities of the holdout targets are the highest probabilities during  
  decoding. This criterion ensures that the Solver’s output not only matches  
  the expected solution but also reflects a confident and correct sequence  
  generation.

### 3.3. Tutor Loss

The Tutor's role is to generate problem instances that effectively train the  
Solver. The Tutor’s loss function is more complex, combining elements of  
supervised learning, reinforcement learning, and the newly integrated  
continuous constraint loss to optimize the generation process.

- **Difficulty Loss**: The first component of the Tutor’s loss is a difficulty  
  loss, calculated using binary cross-entropy. This loss measures how close  
  the generated examples are to the target difficulty, typically set around a  
  50% success rate for the Solver. The Tutor aims to produce examples that are  
  neither too easy nor too hard, and the difficulty loss ensures that the  
  generated examples are appropriately challenging.

- **Diversity Control Loss**: The Tutor is the only component that is  
  trained with a diversity control loss. This loss guides the Tutor’s  
  generation process, ensuring that the examples it produces are within the  
  desired difficulty range and adhere to the necessary level of speculation. The  
  constraint loss works in tandem with the difficulty and reinforcement losses  
  to maintain a balance between generating challenging yet solvable examples  
  and ensuring they remain relevant to the ARC-AGI tasks.

- **Reinforcement Loss**: The second component is a reinforcement-based loss.  
  This loss is derived from the validation performance of the Solver on the  
  generated examples. Here’s how it works:
  
  - **Batch Generation and Caching**: The Tutor generates batches of examples  
    during training. For each batch, the associated validation loss is  
    calculated after the Solver has been trained on these examples. Each  
    generation step caches the Tutor's generated examples (targets) along with  
    the associated validation loss.
  
  - **Reinforcement Step**: After every N batches, a reinforcement learning  
    step is performed. The past N batches are reviewed, and the batch with the  
    lowest validation loss is selected. The Tutor is then trained using  
    cross-entropy loss with the targets from the best-performing batch. This  
    process reinforces the Tutor’s generation strategy by encouraging it to  
    produce examples similar to those that lead to the best validation  
    performance.


## 4. Considerations and Enhancements

### Smart Label Smoothing

Label smoothing should probably be applied in a smart manner. For instance, 
it would make sense to have a higher label smoothing.
constant on inputs than outputs, since inputs are mapped to outputs.

### Dynamic Label Smoothing.

- **Dynamic Adjustment**: The continuous constraint loss approach allows for  
  dynamic scheduling of label smoothing levels throughout the training  
  process. By varying the smoothing levels, the model can alternate between  
  more explorative phases (higher smoothing) and consolidation phases (lower  
  smoothing). This adaptability helps ensure that the Tutor remains flexible  
  and capable of generating a diverse range of examples.

### Balance between constraint and difficulty loss

Ideally, we would like the tutor to exist in a state in which it is mostly fully trained
in terms of the constraint loss, and the difficulty loss encourages exploration of perturbations
of these constraints. An adiabatic pertubation is then applied by the scheduling to bring
us from an exploration to an explatation phase, or back again

However, scheduling that is too rapid will destroy this equilibrium.

### Reinforcement learning of diversity and difficulty

Reinforcement learning can likely also be applied to tutor difficulty
and tutor diversity. However, it will tend to be something that will need 
to be given a large number of batches before evaluating any effect, since 
the teaching distribution would need time to change it's targets.

## Conclusion

This approach effectively guides the Tutor's learning process by directly  
leveraging the validation performance of the Solver and the Ranker’s feedback.  
It ensures that the Tutor generates examples that improve the Solver's  
generalization ability, balancing standard supervised learning, reinforcement  
learning, and continuous constraint loss to tackle the ARC-AGI challenge.
