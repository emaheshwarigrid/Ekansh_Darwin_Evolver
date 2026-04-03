# Batch Size vs Mutation Effectiveness for Failure-Case-Driven Evolution in Darwinian Evolver
## Overview: How Failure-Case Batch Size Enters the Evolver
Darwinian Evolver exposes a tunable `batch_size` hyperparameter that controls how many failure cases are supplied to mutators for each parent organism in an iteration. This batch size is realized via per-parent sampling of trainable failure cases and is tracked as an *effective* batch size at runtime through `EvolverStats.effective_batch_size`.
## Code Evidence: Tracking Effective Failure-Case Batch Size in EvolverStats
[evolver.py L20-L33](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L20-L33)
```python
class EvolverStats(BaseModel):
    num_mutate_calls: int = 0
    num_failure_cases_supplied: int = 0

    @computed_field
    def effective_batch_size(self) -> float:
        if self.num_mutate_calls == 0:
            return 0.0
        return self.num_failure_cases_supplied / self.num_mutate_calls
```
`EvolverStats` computes an *effective* batch size as the average number of failure cases supplied per mutator call, rather than assuming that the configured `batch_size` is always achieved. This definition already anticipates that larger configured batches may not fully materialize (for example, if there are fewer available failure cases of a given type), which is important when reasoning about trade-offs between configured and realized batch sizes.
## Code Evidence: Wiring the Failure-Case Batch Size into the Evolver Loop
[evolver.py L84-L112](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L84-L112)
```python
def __init__(..., batch_size: int = 1, ...) -> None:
    # The number of failure cases that we make available to mutators
    # for a given parent organism.
    assert batch_size > 0, "Batch size must be positive"
    self._batch_size = batch_size
```
The constructor comment explicitly defines `batch_size` as *“The number of failure cases that we make available to mutators for a given parent organism”*, making it a first-class evolutionary hyperparameter rather than a low-level implementation detail. The assertion that `batch_size > 0` guarantees that every mutator call is anchored around at least one concrete failure, preventing degenerate “uninformed” mutations.
## Code Evidence: Per-Parent Failure-Case Sampling and Mutator-Specific Batch Realization
[evolver.py L123-L152](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L123-L152)
```python
for organism, evaluation_result in parents:
    failure_cases = evaluation_result.sample_trainable_failure_cases(batch_size=self._batch_size)
    for mutator in self._mutators:
        failure_cases_for_mutator = (
            failure_cases if mutator.supports_batch_mutation else [failure_cases[0]]
        )
        num_mutate_calls += 1
        num_failure_cases_supplied += len(failure_cases_for_mutator)
```
For each sampled parent, the evolver selects a batch of failure cases using the configured `batch_size`, then *adapts* that batch per mutator based on each mutator’s `supports_batch_mutation` flag. Non-batching mutators see only a single failure (the first in the batch), while batch-capable mutators are given the entire sampled batch, meaning the realized effective batch size depends both on the problem’s failures and the mutator mix.
## Code Evidence: Failure-Type-Aware Mini-Batch Sampling
[problem.py L118-L151](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problem.py#L118-L151)
```python
def sample_trainable_failure_cases(self, batch_size: int = 1) -> list[EvaluationFailureCase]:
    """All failure cases in the resulting batch will be of the same failure_type."""
    failure_type = random.choices(
        list(failure_type_frequencies.keys()),
        weights=list(failure_type_frequencies.values()),
        k=1,
    )[0]
    failure_cases_of_type = [
        failure_case for failure_case in self.trainable_failure_cases if failure_case.failure_type == failure_type
    ]
    effective_batch_size = min(batch_size, len(failure_cases_of_type))
    return random.sample(failure_cases_of_type, effective_batch_size)
```
`EvaluationResult.sample_trainable_failure_cases` enforces that all failure cases in a batch share the same `failure_type`, and caps the realized batch size by the number of available failures of that type. This design intentionally creates *focused mini-batches* around a single failure mode rather than a global batch mixing heterogeneous failures, which is central to the trade-off considered in this question.
## Code Evidence: Mutator-Level Support for Batch vs Single-Case Mutations
[problem.py L177-L201](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problem.py#L177-L201)
```python
@abstractmethod
def mutate(
    self,
    organism: OrganismT,
    failure_cases: list[EvaluationFailureCaseT],
    learning_log_entries: list[LearningLogEntry],
) -> list[OrganismT]:
    """
    The failure_cases list will have size exactly 1 if
    supports_batch_mutation is False, or at least 1 if
    supports_batch_mutation is True.
    """
```
[problem.py L203-L210](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problem.py#L203-L210)
```python
@property
def supports_batch_mutation(self) -> bool:
    """If True, the `mutate` method can accept multiple failure cases at once."""
    return False
```
The mutator interface documents that `failure_cases` is guaranteed to be of size exactly one when batch mutation is disabled, and of size at least one when enabled, thereby defining the contract under which per-failure or mini-batch-style gradient-like signals are delivered to LLM-based mutators. The default `supports_batch_mutation = False` makes single-failure, highly focused mutations the baseline behavior, and batch mutation an explicit opt-in.
## Architectural Observation: Focused Mini-Batches vs Global Failure Coverage
Imbue’s design combines three key ideas: (1) a global `batch_size` hyperparameter per parent, (2) per-type sampling to ensure all failures in a batch share a failure mode, and (3) mutator-level opt-in for batch processing of failures. This architecture biases the system toward *focused* mini-batches that form strong, coherent signals for LLM mutators, rather than a single, large batch that attempts to cover all observed failures simultaneously.
## Theoretical Background: Batch Mutations as Mini-Batch Optimization
The research article describes batch mutations explicitly as analogous to mini-batches in stochastic gradient descent (SGD): rather than mutating against a single failure example, the mutator can see several failure points at once and attempt a change that improves multiple cases. In SGD, there is a well-known trade-off: larger batches reduce gradient noise but can harm generalization and increase per-step cost, while smaller batches introduce noise that can help exploration but require more steps.

In the evolver, mutator calls play a role analogous to gradient steps, and failure cases within a batch are analogous to samples in a mini-batch: each failure exposes a local direction in which the organism needs to improve. A small, coherent mini-batch provides a strong but fine-grained direction ("fix this specific failure mode"), while an overly large or heterogeneous batch can blur the signal by mixing many competing directions, especially within the finite context window of an LLM.
## Mathematical Formalization: Effective Batch Size and Failure-Type Sampling
Within a single evolver iteration, let:

- $N_m$ = `num_mutate_calls`
- $N_f$ = `num_failure_cases_supplied`

The effective batch size is

$$
b_{\text{eff}} = \frac{N_f}{N_m}.
$$

This is precisely what `EvolverStats.effective_batch_size` computes. If the configured `batch_size` is $B$, but some parents have fewer available failures for the sampled type, or many mutators do not support batch mutation, then $b_{\text{eff}}$ can be substantially lower than $B$.

For `EvaluationResult.sample_trainable_failure_cases`, define:

- $T$ = set of failure types
- $n_t$ = unweighted count of trainable failures of type $t \in T$
- $w_t$ = `failure_type_weights[t]` (default 1.0 when not overridden)
- $f_t = n_t w_t$ = weighted frequency for type $t$

The probability of selecting failure type $t$ is

$$
P(\text{type} = t) = \frac{f_t}{\sum_{s \in T} f_s}.
$$

Once a type $t$ has been chosen, the method samples up to $B$ failures of that type; if there are $n_t$ failures of type $t$, then the realized batch size for that parent and type is

$$
b_t = \min(B, n_t).
$$

Thus the expected number of failure cases per mutator call (from the evaluator’s perspective) is

$$
\mathbb{E}[b] = \sum_{t \in T} P(\text{type} = t) \cdot b_t.
$$

This expectation increases with $B$ but saturates once $B$ exceeds the typical $n_t$ for frequently sampled types, which limits the benefits of very large batch sizes even before considering LLM context constraints.
## Mathematical Reasoning: Why Smaller, Focused Batches Can Be More Effective
Consider a batch-capable mutator that receives a set of $b$ failures of a single type and attempts to produce a child that resolves as many of them as possible. Let $p_b$ be the probability that a single mutator call yields a *useful* mutation (for example, improves at least one of the batched failures without regressing too many others).

Two opposing forces shape $p_b$:

- **Signal strength:** As $b$ increases, the mutator sees more varied examples of the same failure type, improving its ability to infer a general pattern and propose a structural fix, which tends to *increase* $p_b$.
- **Context and cognitive load:** As $b$ increases, the prompt grows longer and more complex, reasoning becomes harder, and interactions between constraints can become contradictory ("fix all these at once" may require complex refactoring), which tends to *decrease* $p_b$ beyond some point.

There will typically be an intermediate batch size $b^*$ that maximizes $p_b$, and this optimum can be well below the number of available failures, especially with LLMs constrained by context length and reasoning robustness.

The *throughput* of useful mutations per unit time can be approximated as

$$
\text{throughput}(b) \approx \frac{p_b}{c_b},
$$

where $c_b$ is the average cost (tokens, runtime, and dollars) of a mutator call with batch size $b$. Because $c_b$ tends to grow at least linearly with $b$ while $p_b$ saturates or even declines beyond $b^*$, smaller, focused batches can yield higher throughput of successful mutations.
## Numeric Example: Comparing Small vs Large Batches
Consider a single failure type with $n_t = 12$ trainable failures available for a parent, and assume:

- Configured `batch_size` $B \in \{1, 4, 12\}$
- Mutator supports batch mutation
- Each mutator call costs $c_b$ units of time proportional to $b$

Assume the following (stylized) empirical relationship between batch size and success probability per call:

- $b = 1$: The mutator focuses on a single, sharp example; success probability $p_1 = 0.30$; cost $c_1 = 1$.
- $b = 4$: The mutator sees four representative failures of the same type; success probability increases to $p_4 = 0.50$; cost $c_4 = 4$.
- $b = 12$: The mutator sees all failures at once; the prompt becomes long and entangled, making reasoning harder; success probability drops to $p_{12} = 0.35$; cost $c_{12} = 12$.

From [^8], estimated useful-mutation throughput (probability of success per unit cost) is:

- $\text{throughput}(1) \approx 0.30 / 1 = 0.30$.  
- $\text{throughput}(4) \approx 0.50 / 4 = 0.125$.  
- $\text{throughput}(12) \approx 0.35 / 12 \approx 0.029$.

In this scenario, a batch size of 1 actually achieves the highest rate of useful mutations per unit cost, despite seeing fewer failures per call. A moderate batch size of 4 produces more robust, pattern-based improvements but at significantly lower throughput. A very large batch (12) underperforms both smaller options due to prompt bloat and cognitive overload, even though it nominally "covers" all failures at once.

The core lesson is that *coverage per call* is not the right objective; *useful improvements per unit cost* is. Smaller, focused batches can win under that metric.
## Design Reasoning: Why the Repository Biases Toward Focused Mini-Batches
Several design decisions in Darwinian Evolver suggest that the authors expect small, focused failure batches to often be more effective than large, exhaustive ones:

1. **Per-type coherence over global coverage.** `sample_trainable_failure_cases` always selects failures of a single `failure_type`, even when many heterogeneous failures exist. This choice prioritizes coherent signals for the mutator over simultaneous coverage of all observed failures, implicitly accepting that different failure modes are best attacked in separate mutations.
2. **Default single-failure behavior.** The baseline mutator contract has `supports_batch_mutation = False`, guaranteeing `failure_cases` length 1 for the majority of mutators unless they explicitly opt in to batch processing. This makes "single sharp failure per mutation" the default evolutionary step.
3. **Global batch size but per-mutator adaptation.** `Evolver` uses a single `_batch_size` but downgrades it to 1 for mutators that do not support batching, and may not achieve it when there are few failures of the sampled type. This again suggests that the system is designed to retain the advantages of small, focused steps even when a larger batch size is configured.
4. **LLM-centric constraints and prior experience.** The accompanying research article emphasizes context-length constraints and the difficulty of optimizing LLM-based systems end-to-end, which motivated evolution in the first place. Within such constraints, prompts that attempt to describe all remaining failures at once would likely be unwieldy, motivating a bias toward smaller, information-dense batches.

Taken together, these choices implement a form of *structured stochastic search* where each mutator call focuses on a single failure mode, and batch size controls the number of independent examples of that mode, not the number of distinct failure types.
## Why Small, Focused Batches Can Outperform Large Global Batches Covering All Failures
### 1. Signal Quality vs. Signal Dilution
Providing the mutator with a small number of tightly related failures of the same type produces a clean, high-SNR (signal-to-noise ratio) learning signal: the mutator can look for a pattern that explains all examples and implement a structural fix. In contrast, a large batch that attempts to cover all remaining failures would mix many unrelated failure patterns, making it harder for the LLM to infer a single coherent change that improves them all, often leading to over-complicated or brittle edits.
### 2. Context Window and Prompt Complexity
LLMs have finite context windows and performance that degrades as prompts become long, repetitive, or internally contradictory. A small batch with 1–4 failures keeps the failure description compact and cognitively manageable, leaving room in the prompt for the parent’s code, learning log entries, and mutation instructions. A large batch that attempts to cover all failures may exceed practical context limits or force the mutator to truncate or summarize failures, weakening the mutation signal.
### 3. Exploratory Diversity Through Multiple Mutator Calls
The evolver samples parents stochastically and invokes multiple mutators in parallel per parent, leading to many independent mutation attempts over time. A strategy of small, focused batches allows different calls to explore different failure types or different sub-patterns within the same type across iterations, improving global exploration. Large, exhaustive batches concentrate many failures into a single call, reducing the number of independent "shots on goal" that the search process can take.
### 4. Interaction with Post-Mutation Verification
The research article recommends an optional post-mutation verification step that re-evaluates a mutation only on the (few) failure cases that were passed into the mutator and discards mutations that do not improve any of them. With small batches, this mini-evaluation is cheap and sharply predictive: if a mutation cannot fix even one or two carefully chosen failures of a given type, it is unlikely to be a promising global solution. With large batches, verification becomes more expensive and potentially less informative (a mutation might help some failures and hurt others, leading to ambiguous signals), lowering the filter’s efficiency.
### 5. Avoiding Overfitting to a Narrow Subset While Still Generalizing
Darwinian Evolver separates the training (mutator feedback) set from the scoring set: mutators see only failure cases drawn from a training subset, while fitness is computed on a broader scoring dataset. Small batches encourage the mutator to find changes that robustly fix a few representative failures, which is more likely to generalize to unseen scoring cases of the same type. A large batch covering all observed training failures can tempt the mutator to over-optimize for that specific finite set (for example, by hard-coding cases), potentially harming performance on the scoring set and thus evolutionary fitness.
## Concrete Example: Focused vs Exhaustive Failure Coverage Over Iterations
Consider a parent organism with three failure types:

- Type A: 8 failures
- Type B: 4 failures
- Type C: 2 failures

Assume `failure_type_weights` are all 1.0, and a batch-capable mutator is used.
### Case 1: Small, Focused Batches ($B = 2$)
Per parent, `sample_trainable_failure_cases` chooses a type with probability proportional to its count:

- $P(A) = 8 / (8+4+2) = 8/14 \approx 0.57$.
- $P(B) = 4 / 14 \approx 0.29$.
- $P(C) = 2 / 14 \approx 0.14$.

Once a type is chosen, the method samples up to two failures of that type. Over many iterations:

- Multiple independent mutations will target type A with different pairs of failures, exploring diverse improvements for that mode.
- Type B and C still receive attention proportional to their frequency, but through many separate calls rather than a single giant batch.

Each mutation focuses on a pair of highly related failures, keeping prompts compact and allowing the mutator to propose clean, structural fixes.
### Case 2: Large Batch Covering All Failures (Conceptual $B = 14$)
If the evolver instead tried to pass all failures at once (ignoring its per-type constraint), a mutator would receive 14 heterogeneous failures in a single call. The mutator would need to:

- Parse and understand many distinct failure contexts.
- Infer a change that simultaneously fixes all of them.
- Avoid regressions on already-passing cases.

In practice, the mutator might:

- Produce a very complex patch that tries to handle many special cases, increasing the chance of new bugs.
- Focus on the most salient failures and ignore others, effectively wasting some of the supplied information.

Meanwhile, the evolver would have consumed a large amount of context and tokens for a *single* mutation attempt that may not outperform several smaller, focused attempts spread across iterations.

This illustrates why the repository’s design chooses to focus on per-type mini-batches and exposes `batch_size` as a tuning knob for *examples per failure mode*, not for "fix everything at once" coverage.
## ASCII Sequence Diagram: Flow of Failure-Case Batching and Mutation
```text
+------------------+      +------------------------+      +---------------------------+
| Population       |      | Evolver               |      | EvaluationResult          |
| (organisms +     |      | (per iteration)       |      | (per parent)              |
|  scores)         |      |                       |      |                           |
+---------+--------+      +-----------+-----------+      +-------------+-------------+
          |                           |                          |
          | sample_parents()          |                          |
          +-------------------------> |                          |
                                      |  for each (organism,    |
                                      |     evaluation_result): |
                                      |                          |
                                      |  sample_trainable_      |
                                      |  _failure_cases(        |
                                      |      batch_size=B )     |
                                      +------------------------>+
                                                                 |
                                                                 | choose failure_type t
                                                                 | compute effective_batch_size
                                                                 | return list[FailureCase]_t
                                      +<------------------------+
                                      |
                                      | for mutator in mutators:
                                      |   if mutator.supports_batch_mutation:
                                      |       failure_cases_for_mutator = batch
                                      |   else:
                                      |       failure_cases_for_mutator = [batch]
                                      |
                                      |   submit mutate(organism,
                                      |                 failure_cases_for_mutator,
                                      |                 learning_log_entries)
                                      v
                            +---------+-----------+
                            | Mutator            |
                            | (LLM-powered)      |
                            +---------+----------+
                                      |
                                      | mutated_organisms
                                      v
                            +---------+-----------+
                            | Evaluator          |
                            | (verify + score)   |
                            +---------+----------+
                                      |
                                      | add(mutated_organism,
                                      |     evaluation_result)
                                      v
                             +--------+---------+
                             | Population       |
                             +------------------+
```

This diagram highlights that `batch_size` is realized at the `EvaluationResult` level (per-type mini-batch selection) and then modulated per mutator via `supports_batch_mutation`, before flowing through verification and scoring back into the population.
## Architectural Alternatives: Other Ways to Exploit Failure Cases
Several alternative designs could have been implemented, each with different trade-offs relative to the current focused mini-batch approach:

1. **Global, heterogeneous batches per parent.** Instead of enforcing a single `failure_type` per batch, the evolver could randomly sample failures across all types. This would maximize failure-type coverage per call but would send a noisy, heterogeneous signal to the mutator, making coherent improvements harder and exacerbating context issues.
2. **Adaptive batch sizing per type.** The system could adapt `batch_size` dynamically based on observed mutator success rates per failure type (for example, increasing $b$ for stubborn failure types and reducing it when large batches hurt performance). While potentially powerful, this would add complexity and make the system harder to reason about and tune.
3. **Curriculum-style batching.** The evolver could begin with small batches (or even single failures) to explore the space, then gradually increase batch size for failure types that appear frequently, analogous to curriculum learning. This might accelerate convergence once strong candidate solutions exist but again risks overfitting and context overload.
4. **Hierarchical batching across parents.** Another design could aggregate failures of the same type across multiple parents into a single, cross-parent batch mutation. While this might exploit shared structure between different lineages, it would complicate lineage tracking, post-mutation verification, and the learning log semantics.

The current architecture opts for a simpler, robust design that emphasizes per-type focus, tunable mini-batch size per mutation, and high-throughput exploration via many relatively cheap mutator calls. Within this design, small batches of focused failures often outperform large, all-encompassing batches because they produce stronger, more coherent signals for LLM mutators under real-world constraints of context, cost, and evolutionary search dynamics.

---
