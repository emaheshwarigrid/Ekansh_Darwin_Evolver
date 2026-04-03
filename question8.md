# Why Darwinian Evolver Uses User-Defined Evaluators

## Overview: Evaluation as a Pluggable, Problem-Specific Component

Darwinian Evolver treats evaluation as a user-supplied component so that the same evolutionary core can optimize prompts, code, agents, or arbitrary text under domain-specific metrics, rather than being hard-wired to a single scoring protocol. The framework only assumes that an evaluator can map an organism to a scalar score and structured failure cases; how that score is computed, which data are used, and what failures mean are entirely up to the integrator. This separation is what allows the authors to describe the system as a “near-universal optimizer” that can operate on “near arbitrary code and text problems” as long as a scoring procedure exists.

## Observation: Evaluation Is Delegated to User-Supplied Evaluator Objects

The core `Problem` abstraction explicitly requires an `evaluator` field that is an instance of a user-implemented `Evaluator` subclass, not a reference to any built-in evaluation routine. The abstract `Evaluator` base class defines only an interface (`evaluate`, optional `verify_mutation`, and optional output destinations), with no default logic for computing scores, datasets, or metrics. Example problems such as `parrot` construct problem instances by instantiating their own evaluator classes and passing them into `Problem`, demonstrating that each task defines its own evaluation logic.

## Code Evidence: Problem and Evaluator APIs Enforce User-Defined Evaluation

### Problem composition requires a user-specified evaluator

[problem.py L239-L245](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/problem.py#L239-L245)
```py
class Problem(BaseModel, Generic[OrganismT, EvaluationResultT, EvaluationFailureCaseT]):
    initial_organism: OrganismT
    evaluator: Evaluator[OrganismT, EvaluationResultT, EvaluationFailureCaseT]
    mutators: list[Mutator[OrganismT, EvaluationFailureCaseT]]
```
This shows that every `Problem` instance must be constructed with a concrete `Evaluator` instance; there is no default evaluator attached to the framework itself, and the generic type parameters make the evaluator’s organism and result types problem-specific.

The README mirrors this contract at the user API level.

[README.md L39-L50](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/README.md#L39-L50)
```py
A Problem is composed of the following:
class Problem(Generic[OrganismT, EvaluationResultT, EvaluationFailureCaseT]):
    initial_organism: OrganismT
    evaluator: Evaluator[OrganismT, EvaluationResultT, EvaluationFailureCaseT]
    mutators: list[Mutator[OrganismT, EvaluationFailureCaseT]]
```
The public documentation makes clear that “to adapt Darwinian-evolver to your own use case, you need to define a `Problem`” and that the evaluator is one of the three required user components (initial organism, evaluator, mutator).

### Evaluator base class is abstract and domain-agnostic

[problem.py L213-L220](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/problem.py#L213-L220)
```py
class Evaluator(ABC, Generic[OrganismT, EvaluationResultT, EvaluationFailureCaseT]):
    @abstractmethod
    def evaluate(self, organism: OrganismT) -> EvaluationResultT:
        """Evaluate the organism and return a fitness score."""
        raise NotImplementedError("Evaluators must implement the evaluate method")
```
Here `Evaluator` is an abstract base class with a single required method, `evaluate`, which must return an `EvaluationResult` subclass defined by the problem author. There is no reference to datasets, loss functions, accuracy, or any particular evaluation metric, which forces each problem to define what “fitness” means.

The optional `verify_mutation` hook similarly has no default semantics:

[problem.py L222-L228](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/problem.py#L222-L228)
```py
    def verify_mutation(self, organism: OrganismT) -> bool:
        """
        Verify that the mutation of the organism has addressed the given failure cases (or some fraction of them).

        This method is optional for evaluators to implement.
        """
        raise NotImplementedError("This evaluator does not support mutation verification")
```
By default, verification is unsupported and must be supplied by the evaluator, reinforcing that even the notion of “mini-evaluation” is problem-specific.

### Example: `parrot` problem defines a bespoke evaluator

[parrot.py L115-L169](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/problems/parrot.py#L115-L169)
```py
class ParrotEvaluator(Evaluator[ParrotOrganism, EvaluationResult, ParrotEvaluationFailureCase]):
    def evaluate(self, organism: ParrotOrganism) -> EvaluationResult:
        trainable_failure_cases = []
        holdout_failure_cases = []
        for i, phrase in enumerate(self.TRAINABLE_PHRASES):
            response = organism.run(phrase)
            if response != phrase:
                trainable_failure_cases.append(
                    ParrotEvaluationFailureCase(
                        phrase=phrase,
                        response=response,
                        data_point_id=f"trainable_{i}",
                    )
                )
        num_total = len(self.TRAINABLE_PHRASES) + len(self.HOLDOUT_PHRASES)
        num_correct = num_total - len(trainable_failure_cases) - len(holdout_failure_cases)
        score = num_correct / num_total
        is_viable = num_correct > 0

        return EvaluationResult(
            score=score,
            trainable_failure_cases=trainable_failure_cases,
            holdout_failure_cases=holdout_failure_cases,
            is_viable=is_viable,
        )
```
This evaluator chooses a very specific protocol: compare LLM output to target phrases, derive a success-rate score, and treat any failure as a per-phrase `EvaluationFailureCase` that is either trainable or holdout. Nothing in the Evolver core encodes these choices; they are entirely encapsulated in the evaluator.

The problem factory wires this evaluator into a `Problem` instance:

[parrot.py L180-L186](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/problems/parrot.py#L180-L186)
```py
def make_parrot_problem() -> Problem:
    return Problem[ParrotOrganism, EvaluationResult, ParrotEvaluationFailureCase](
        evaluator=ParrotEvaluator(),
        mutators=[ImproveParrotMutator()],
        initial_organism=ParrotOrganism(prompt_template=INITIAL_PROMPT_TEMPLATE),
    )
```
This pattern repeats across other problems (ARC-AGI variants, circle packing, multiplication verification), confirming that every concrete use-case supplies its own evaluator class with custom data, scoring, and failure semantics.

### Evolver and population depend only on evaluator outputs

The main evolutionary loop is parameterized by an `Evaluator` instance and never inspects datasets, test cases, or metrics directly.

[evolver.py L84-L90](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/evolver.py#L84-L90)
```py
class Evolver:
    _population: Population
    _learning_log_view: LearningLogView
    _mutators: list[Mutator]
    _evaluator: Evaluator
```

Within a generation, evaluation is always routed through the evaluator object:

[evolver.py L171-L177](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/evolver.py#L171-L177)
```py
for future in concurrent.futures.as_completed(mutated_organisms_futures):
    organism, should_evaluate = future.result()
    if should_evaluate:
        num_mutations_after_verification += 1
        evaluation_future = evaluator_executor.submit(self._evaluator.evaluate, organism)
        organism_evaluation_futures.append((organism, evaluation_future))
        num_evaluate_calls += 1
```
The Evolver only cares that `evaluate` eventually returns an `EvaluationResult`; it does not know what this entails in terms of problem-specific logic.

The population then interprets evaluation results purely through the generic `score` and `failure_cases` fields, again without knowledge of any fixed evaluation protocol.

[population.py L238-L247, L315-L323](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/population.py#L238-L247-L323)
```py
class WeightedSamplingPopulation(Population):
    _sharpness: float
    _fixed_midpoint_score: float | None
    _midpoint_score_percentile: float | None
    _novelty_weight: float
```
Only the abstract `EvaluationResult.score` and `EvaluationResult.trainable_failure_cases` properties influence parent sampling, leaving the details of how those values were computed entirely up to the evaluator.

## Design Reasoning: Why Evaluation Is Externalized Instead of Fixed

### Universal optimizer goal requires arbitrary scoring functions

The accompanying research post defines Darwinian Evolver as a near-universal optimizer that can operate on “near arbitrary code and text problems” as long as there exists *some* way to score candidate solutions, even approximately. It explicitly emphasizes that the scoring methodology may vary widely: datasets with ground truth, direct measurements of performance (speed, resource use), or heuristic quality metrics derived via code inspection or LLM critiques.

Encoding a fixed evaluation protocol (for example, supervised accuracy on a labeled dataset of input-output pairs) into the core would contradict this goal, as many target problems do not fit that mold:

- Safety or alignment tasks where “correctness” involves satisfying nuanced behavioral constraints rather than matching a single label.
- Agentic workflows where evaluation involves multi-step tool use, external services, and human-in-the-loop approvals.
- Optimization of non-differentiable objectives such as runtime, memory footprint, or qualitative code clarity, possibly combined into composite scores.

By forcing users to define their own evaluator, the framework remains compatible with any problem where organisms can be executed or inspected and assigned a scalar fitness.

### Domain knowledge and constraints live inside evaluators

Different domains require different test harnesses, datasets, and notions of failure, all of which are encoded inside evaluator implementations like `ParrotEvaluator`. For example:

- In the `parrot` task, failure means “the model did not repeat this string exactly,” and the evaluator explicitly distinguishes between trainable phrases and holdout phrases to mitigate overfitting.
- In ARC-like visual reasoning tasks, failure cases might include grid inputs and pixel-by-pixel mismatches, with scores reflecting task-wise success rates instead of per-row accuracy.
- In performance-optimization problems, failures might be timeouts or resource limit violations, and scores may combine correctness with runtime penalties.

If the Evolver tried to impose a single protocol for what a “failure” is or how to compute the score, it would need deep knowledge of every domain, undermining its modularity and making it brittle to new kinds of tasks.

### Robustness to noisy and approximate evaluators

The research post notes that the evolver remains effective with “noisy evaluators” as long as they can approximately rank candidates by quality. Many LLM-centric evaluations are inherently stochastic: the same organism evaluated twice on an LLM-based test harness may yield slightly different scores. By delegating evaluation to user code, the system allows:

- Averaging over multiple runs.
- Incorporating adversarial or stress-test cases.
- Using heuristics (e.g., LLM rubric scores) that are not strictly repeatable.

A fixed, built-in protocol would either have to assume determinism (which fails in practice) or be so generic that users would still have to wrap it in custom logic; making evaluation user-defined avoids this tension.

### LLM and infrastructure constraints are problem-dependent

Evaluation often interacts with external APIs (LLM providers), internal services, and cost constraints that differ between deployments. The `parrot` evaluator, for instance, calls Anthropic’s API with a specific model name and parameters, and may need to throttle or batch requests to stay within rate limits. Other evaluators may:

- Use entirely different providers or on-prem models.
- Cache partial results to reduce cost.
- Subsample large datasets per evaluation to fit within context windows.

Centralizing such concerns in a fixed evaluation protocol inside the evolver would tightly couple the framework to specific providers and infrastructure assumptions, making it harder to adapt and evolve.

## Mathematical Role of Evaluator Scores in Selection

### Weight computation from evaluator outputs

The `WeightedSamplingPopulation` converts evaluator-produced scores into sampling weights via a sigmoid transform and novelty bonus.

[population.py L359-L387](https://github.com/imbue-ai/darwinian_evolver/blob/4a5597b55634255610e569c9c5bcc34ec9142998/darwinian_evolver/population.py#L359-L387)
```py
def _compute_weights(self, eligible_organisms, novelty_weight):
    midpoint_score = self._compute_midpoint_score()
    weights = []
    for organism, evaluation_result in eligible_organisms:
        sigmoid_performance = self._compute_sigmoid_performance(evaluation_result, midpoint_score=midpoint_score)
        novelty_bonus = self._compute_novelty_bonus(organism, novelty_weight)
        weight = sigmoid_performance * novelty_bonus
        weights.append(weight)
    return weights

def _compute_sigmoid_performance(self, evaluation_result: EvaluationResult, midpoint_score: float) -> float:
    return 1 / (1 + math.exp(-self._sharpness * (evaluation_result.score - midpoint_score)))
```
This logic depends only on the numeric `evaluation_result.score`, not on how that score was produced. Formally, for organism $i$ with evaluator score $s_i$, midpoint $m$, and sharpness $\lambda$, the performance component is
$$
\sigma_i = \frac{1}{1 + e^{-\lambda (s_i - m)}}.
$$ [^1]

The novelty bonus term is
$$
b_i = \frac{1}{1 + \tau n_i},
$$ 
where $n_i$ is the number of existing children of organism $i$, and $\tau$ is the `novelty_weight`.

The overall unnormalized sampling weight is then
$$
w_i = \sigma_i \cdot b_i.
$$ 

The evaluator’s job is to choose a scoring function $f(\cdot)$ such that $s_i = f(\text{organism}_i)$ encodes the problem’s desiderata (correctness, efficiency, safety, etc.). Different evaluators correspond to different choices of $f$, which can radically change the induced distribution over parents even when the evolver’s selection mathematics is fixed.

### Numeric example: how evaluators alter selection probabilities

Consider three organisms A, B, and C. Assume novelty bonuses are equal $b_i = 1$ (each has the same number of children), and the sampling is dominated by sigmoid-scaled scores.

**Scenario 1: Evaluator uses plain accuracy**

Suppose the evaluator’s score is accuracy on a dataset, yielding

- $s_A = 0.95$,
- $s_B = 0.90$,
- $s_C = 0.70$.

Let midpoint $m = 0.80$ and sharpness $\lambda = 10$ (the README’s default sharpness range). Then

$$
\sigma_A = \frac{1}{1 + e^{-10(0.95 - 0.80)}} \approx \frac{1}{1 + e^{-1.5}} \approx 0.82,
$$ 
$$
\sigma_B = \frac{1}{1 + e^{-10(0.90 - 0.80)}} \approx \frac{1}{1 + e^{-1.0}} \approx 0.73,
$$ 
$$
\sigma_C = \frac{1}{1 + e^{-10(0.70 - 0.80)}} \approx \frac{1}{1 + e^{1.0}} \approx 0.27.
$$ 

The relative weights are approximately $w_A : w_B : w_C \approx 0.82 : 0.73 : 0.27$, so A and B are much more likely to be sampled than C.

**Scenario 2: Evaluator adds a runtime penalty**

Now imagine a different evaluator that trades off accuracy against runtime cost:
$$
s_i' = \text{accuracy}_i - 0.5 \cdot \text{runtime}_i,
$$ 
where runtime is normalized between 0 and 1. Suppose

- A is slow: accuracy $0.95$, runtime $0.8$ $\Rightarrow s_A' = 0.95 - 0.5 \cdot 0.8 = 0.55$,
- B is moderate: accuracy $0.90$, runtime $0.4$ $\Rightarrow s_B' = 0.90 - 0.5 \cdot 0.4 = 0.70$,
- C is fast but less accurate: accuracy $0.80$, runtime $0.1$ $\Rightarrow s_C' = 0.80 - 0.5 \cdot 0.1 = 0.75$.

Keep $m = 0.75$, $\lambda = 10$. Then

$$
\sigma_A' = \frac{1}{1 + e^{-10(0.55 - 0.75)}} \approx \frac{1}{1 + e^{2.0}} \approx 0.12,
$$ [^8]
$$
\sigma_B' = \frac{1}{1 + e^{-10(0.70 - 0.75)}} \approx \frac{1}{1 + e^{0.5}} \approx 0.38,
$$ [^9]
$$
\sigma_C' = \frac{1}{1 + e^{-10(0.75 - 0.75)}} = 0.5.
$$ 

Under this alternative evaluator, organism C becomes the most likely parent, even though it had the lowest raw accuracy in Scenario 1. The only change was in the evaluator’s scoring function $f$, not in the evolver’s selection logic; this illustrates how user-defined evaluators provide strong leverage over the evolutionary trajectory.

In practice, evaluators can implement far more complex $f$, for instance blending dataset accuracy, safety violations, latency, and cost into a single scalar via arbitrary mathematical combinations.

## Alternatives: What a Fixed Evaluation Protocol Would Look Like

A fixed evaluation protocol inside Darwinian Evolver might, for example, require:

- A dataset of $(x, y)$ pairs and a user-supplied `run(x)` method on organisms.
- A baked-in metric such as mean accuracy or mean squared error.
- A fixed notion of failure as “mismatch between $\hat{y}$ and $y$.”

The core would then:

- Call `organism.run(x)` on each $x$ in the dataset.
- Compare results to $y$ under a fixed loss or accuracy function.
- Compute a scalar fitness as a function of these losses.

This design would simplify some use-cases but introduce significant limitations:

- **Limited domain coverage.** Problems without labeled datasets, or where correctness is qualitative or multi-step (e.g., dialogue, tool calling), would not fit.
- **No direct control over multi-objective trade-offs.** Tasks that require balancing correctness against runtime, cost, or safety would have to shoehorn these into the fixed metric.
- **Harder integration with existing evaluation harnesses.** Many teams already have bespoke test runners, simulators, or human-review pipelines; forcing them into a specific schema would increase integration friction.
- **Tight coupling to specific infrastructure.** Any built-in protocol that assumes particular logging formats, APIs, or data layouts would make the core less portable.

The repository’s architecture deliberately avoids this by keeping the evolver and population ignorant of evaluation details and treating the evaluator as the sole authority on what constitutes fitness.

## Trade-offs and Benefits of User-Defined Evaluators

### Trade-offs

- **More work for integrators.** Users must implement an evaluator class, including data loading, execution harnesses, and scoring functions, whereas a fixed protocol could be used out-of-the-box for a narrow class of problems.
- **Risk of poorly designed fitness functions.** Because evaluators are arbitrary, a badly chosen score (e.g., one that is too easy to saturate or misaligned with real-world goals) can cause evolution to stall or overfit.
- **Harder to compare results across problems.** Different evaluators often use incomparable scales, making it non-trivial to aggregate or benchmark across heterogeneous tasks.

### Benefits and resulting generality

Despite these costs, delegating evaluation to user-defined evaluators unlocks several forms of generality:

1. **Domain generality.** Any domain where candidate solutions can be executed or inspected—code, prompts, agent configurations, even text policies—can be optimized, because the core only requires a scalar score and failure cases.
2. **Metric generality.** Evaluators can implement arbitrary scalar functions $f$ that blend correctness, robustness, performance, cost, safety, or any other concern, and can change this function over time as requirements evolve.
3. **Protocol generality.** Some tasks may use dataset-based evaluation; others may rely on simulators, game environments, human feedback, or rule-based checkers. All are expressible as evaluator logic.
4. **Failure-structure generality.** By customizing `EvaluationFailureCase` and `EvaluationResult.sample_trainable_failure_cases`, evaluators can steer which error modes the mutators focus on (e.g., false positives vs. false negatives, specific edge cases).
5. **Cost and resource awareness.** Evaluators can incorporate API limits, compute budgets, and stochastic evaluation strategies (e.g., evaluating on random subsets of data) without the evolver needing to understand these constraints.
6. **Future extensibility.** New scoring methods (e.g., learned critics, advanced simulators) can be adopted by writing new evaluator classes, without changing the evolver or population code.

These benefits are consistent with the project’s stated goal of serving as a “problem-agnostic” optimization framework that can be applied wherever a scoring methodology exists.

## Architectural Flow: From Organism Through Evaluator to Selection

The following ASCII diagram summarizes how user-defined evaluators fit into the overall architecture:

```text
+-------------------+         +---------------------------+
|   Problem         |         |   Evolver                |
|-------------------|         |---------------------------|
| initial_organism  |         | _population              |
| evaluator --------+-------->| _evaluator (user-impl)   |
| mutators          |         | _mutators                |
+-------------------+         +-------------+-------------+
                                              |
                                              | evolve_iteration()
                                              v
                                   +----------+-----------+
                                   |  Population          |
                                   |  (WeightedSampling)  |
                                   +----------+-----------+
                                              |
             sample_parents() using scores    |
               and novelty bonuses            |
                                              v
                            +-----------------+-----------------+
                            | (Organism, EvaluationResult)       |
                            +-----------------+------------------+
                                              |
                       failure_cases          |
                                              v
+-------------------+      mutate()      +-----------------------+
|   Mutator(s)      |------------------->|  Mutated Organisms    |
+-------------------+                     +-----------+----------+
                                                      |
                                                      | evaluate(mutant)
                                                      v
                                      +---------------+----------------+
                                      |  Evaluator (user-defined)      |
                                      |  - runs tests / harnesses      |
                                      |  - computes score              |
                                      |  - constructs failure_cases    |
                                      +---------------+----------------+
                                                      |
                                 EvaluationResult(score, failure_cases)
                                                      |
                                                      v
                                      +---------------+----------------+
                                      |  Population.add()              |
                                      +--------------------------------+
```

In this flow, the evaluators are the only components that know how to run organisms and what constitutes success or failure; all other components treat evaluation as a black box that yields scores and structured failures. This design is what necessitates user-defined evaluator functions and simultaneously grants the framework its broad applicability across diverse optimization problems.

---

