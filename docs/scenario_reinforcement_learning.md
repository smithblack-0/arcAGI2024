# Research Proposal: Autonomous Multimodal Reinforcement Learning for ARC-AGI

## Objective:
Develop an autonomous reinforcement learning algorithm for a multimodal
transformer model capable of generating, solving, and refining logical
scenarios for the ARC-AGI challenge, and deducing the rules that solve
such situations.

The model operates across four distinct phases—**Rule Statement**, **Scenario 
Generation**, **Rules Deduction**, and **Rules Application**—each with distinct
reinforcement objectives. These phases may also have subphases, which will be
discussed in more detail.

The system is designed to be fully autonomous, allowing the model to
independently improve both its **Scenario Generation** and **Solving**
capabilities through reinforcement learning. The **Scenario Creation zones**
(**Rule Statement**, **Scenario Generation**) operate in opposition to the
**Solving Management** sections (**Rules Deduction**, **Rules Application**) in
a reinforcement-based GAN architecture. This creates a dynamic where:

- **Scenario Creation**: Receives higher scores for generating difficult but
  solvable scenarios and is heavily penalized for generating unsolvable or junk
  content.
  
- **Solving Management**: Receives rewards for deducing the correct set of rules
  for the problem, which allow the problem to be successfully solved.

Scenarios are selected for reinforcement based on which behavior is performing
well—either scenario creation or solving. A **Sanity Constraint** ensures that
the Scenario Management process remains logically consistent, serving as a hard
constraint that discards insane or unsolvable scenarios. While it may dominate
when required, it only intervenes to maintain logical integrity. **Diversity
Loss** prevents the model from converging on repetitive solutions, maintaining
its creative exploration of logical rules. Additionally, an **Anchor Loss** is
applied to maintain the model's alignment with the core ARC-AGI tasks, ensuring
that the model does not stray too far from the original logic during autonomous
reinforcement learning.

What is actually being trained by reinforcement is **problem-solving behavior**
versus just the ability to propose the correct answer. This may include isolating
interesting details, proposing and testing hypotheses, and other such tactics.
The actual answer after performing the problem-solving process is still trained
by supervised loss.

---

## Context: Multimodal Block Generation

We are generating information in **multimodal blocks** using a transformer-based
model, following principles of supervised or semi-supervised learning. The
multimodal architecture allows the model to generate and represent ARC-AGI int
grid content, structuring the content into four regions or "zones":
**Rule Statement zone**, **Scenario Generation zone**, **Rules Deduction**, and
**Rules Application zone**.

Each region consists of tokens that may span multiple multimodal blocks, and
content flows through these regions in a pipeline. Some of these zones have
subsections that are extremely important to know about. The notable cases here
are the **Scenario Generation**, **Rule Deduction**, and **Rules Application**
zones.

- The **Scenario Generation** zone has the *Scenario* zone (what an ARC-AGI
  question is, including the test output) and the *Scenario Answer* zone, which
  contains the answer to the ARC-AGI problem.
- The **Rule Deduction** zone has the *Deduction Steps* zone, which leads us
  through the process of figuring out the rules, and the *Rules* zone, which
  states what the rules should be based on the line of logic.
- The **Rules Application** zone has the *Solving Steps* zone, which walks us
  through figuring out the problem, and the *Solution* zone, which is the answer
  to the problem itself.

These distinctions are important due to teacher-forcing activity and scoring,
which occur within the model, and due to the fact that zone context will be
limited when performing attention. In particular, the following limitations
apply:

- The **Scenario Generation** zone will have as context the **Rule Statement**
  zone. This encourages the model to come up with examples that correspond to
  the rules.
- The **Deduction Steps** have as context only the **Scenario** zone, which
  means rules must be deduced based **only** on the provided ARC-AGI scenario.
- The **Rules** zone in **Rule Deduction** has access only to the **Scenario**
  and the **Deduction Steps**, from which it should state the rules. In
  practice, a comparison is performed between the **Rule Statement** and the
  predictions made here, and the answer is teacher-forced unless evaluating.
- The **Solving Steps** zone has access only to the **Scenario** and the
  **Rules**. This ensures it operates according to the recovered rules in its
  attempt to solve the scenario.

This setup plays a critical role in the reinforcement system, as each zone has a
specific purpose within the learning process, ensuring the model generates,
solves, and verifies scenarios autonomously.

---

## Context: Feature Engineering Preparation

In order to properly solve the situation for the ARC-AGI problem, some additional
details are needed, which must be manually created by a machine learning
engineer. This data will then be assembled into anchor supervised learning
targets that act both to warm up the pretrained transformer and to provide anchor
loss when the reinforcement state starts.

### Requirements of Manual Data Creation

Consider an ARC-AGI input-output example set and its test case. The machine
learning engineer needs to take this starting case, consider the logic behind
it, and produce the following additional information. This will be used both to
provide an anchor loss and to warm up the pretrained model to generate the
appropriate pattern of tokens.

---

## Warmup and Reinforcement Phases

Let's talk about how training and generation are performed during the phases of
the training process.

### Warmup Phase

The warmup phase will consist of attempting to predict the example content
indicated above, including the special zone tokens, block modes, and other
important features.

To ensure a pretrained model is able to format results sanely, a warmup phase is
first performed using supervised learning. This involves using the data
mentioned above to perform supervised learning with the pretrained model. The
goal here is to ensure that the pretrained model will generate all the different
signals that are needed to identify and use the various zones.

### Reinforcement Phase

The reinforcement phase keeps the warmup content around for anchor loss purposes
but also starts to perform reinforcement learning. This is done by selecting and
applying generated examples that are highly successful in scenario generation
and, separately, successful at solving.

## Reinforcement Algorithms

### Application of Reinforcement by Losses

Reinforcement learning is applied by treating high-performing generated
scenarios as semi-supervised learning targets, using **next-sequence prediction**
training. This method leverages standard transformer techniques to handle
reinforcement losses. However, different portions of the model influence
reinforcement loss depending on performance:

- **Scenario Management Reinforcement**: If the **Scenario Management**
  mechanism performs well, only the **Rule Statement** and **Scenario Generation**
  zones are involved in reinforcement.
- **Solving Reinforcement**: If the two **Solving Management** portions of the
  model perform well, then reinforcement affects the **Rules Deduction** and
  **Rules Application** zones.

---

### Reinforcement Scores

Reinforcement scores for the **Solution Process** and **Scenario Management
zones** are calculated separately to identify which examples should be used for
reinforcement. Here’s how each score is computed:

1. **Solution Score**:
   - The **Solution Score** is a penalty that is based on how effective the
     **Deduction Steps** and **Solving Steps** are at leading the model to the
     correct **Rules** and **Solution** respectively. When the cross-entropy of
     the **Rules** and **Solution** zones is low, the penalty is low.

2. **Scenario Management Score**:
   - The **Scenario Management score** is based on a combination of the
     **Sanity Score** and the **Solver Score**. The **Solver Score** is used to
     drive reinforcement, while the **Sanity Score** ensures that the scenario
     remains solvable.

---

## Additional Losses During Reinforcement

1. **Answer Loss**:
    - Answer loss is designed to speed up convergence. One answer loss comes
      from the cross-entropy between the predicted and actual rules.
    - **Answer loss** is only applied if the **Sanity Penalty** is zero.

2. **Diversity Loss**:
   - **Diversity loss** is critical to preventing **mode collapse**. It is
     calculated by comparing two scenarios: one generated with teacher-forcing
     and one generated without context.

3. **Anchor Loss**:
   - **Anchor loss** continues the **warmup strategy** during reinforcement
     learning. It ensures that the model remains tethered to the core ARC-AGI
     tasks by periodically applying the warmup loss.

## Model Execution: Training, Generation, and Evaluation

The model operates through different steps depending on whether it is in training,
generation, or evaluation mode. Each mode involves different dynamics based on how
zones interact.

### Training
During training, the model uses supervised learning to predict correct sequences
across the various zones. The focus is on learning patterns that connect rule
statements, scenarios, deduction steps, solving steps, and solutions.

- **Next Sequence Prediction**: The model uses warmup data and performs next
  sequence prediction. Teacher-forcing is applied to guide the model toward
  correct sequences.
- **Supervised Learning**: All zones are used in supervised learning, where the
  model is trained to predict correct outputs in each zone.

### Generation Process
When generating examples, the model follows a step-by-step process where some
zones are decoded using sampling, while others use teacher-forcing. The process
is as follows:

1. **Rule Statement Generation**: The **Rule Statement** is generated using
   decoded sampling.
2. **Scenario Generation**: The **Scenario** is generated via decoded sampling.
3. **Deduction Steps**: The model generates the **Deduction Steps** via sampling
   but does **not** generate the **Rules**.
4. **Rules Loss and Accuracy**: The loss between the predicted rules and the
   actual rules is computed. The rules are generated by teacher-forcing with the
   **Rule Statement**, and rule accuracy is saved as a metric.
5. **Solving Steps**: The **Solving Steps** are generated using decoded sampling
   to apply the rules.
6. **Solution Handling**: The **Solution** is teacher-forced using the **Scenario
   Answer** from **Scenario Generation**. Cross-entropy and pixel correctness are
   computed.

At the end of generation, a chain of predicted elements is produced, and scores
such as losses and accuracy are computed for reinforcement.

### Evaluation Process
During evaluation, the process is adjusted to solve the ARC-AGI examples, starting
from **Scenario Generation**:

1. **Scenario Generation**: The process starts with **Scenario Generation**, using
   an ARC-AGI example as input. The **Scenario Answer** is left empty.
2. **Deduction Steps and Rule Sampling**: The **Deduction Steps** are generated,
   and the **Rules** are sampled instead of teacher-forced.
3. **Solving Steps**: The **Solving Steps** are generated based on the sampled
   rules and scenario.
4. **Solution Sampling**: The final **Solution** is generated by sampling the most
   likely prediction.

During evaluation, the model relies on its learned abilities to deduce rules and
solve problems autonomously, without teacher-forcing.

By structuring generation and evaluation in this way, the model can generate and
evaluate examples, allowing it to refine its problem-solving abilities through
reinforcement and supervised learning.
