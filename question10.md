# Mutator Independence & Heterogeneous Improvement Strategies in `darwinian_evolver`

**Architectural Audit Report — `imbue-ai/darwinian_evolver`**
*Principal Systems Architect & Evolutionary AI Researcher Perspective*

***

## 1. Observation: Polymorphic Mutator Contract in the Evolutionary Loop

The `darwinian_evolver` framework treats every improvement strategy as an interchangeable unit by enforcing a single, narrow behavioral contract — the `Mutator` abstract base class. The `Evolver` engine holds a `list[Mutator]` and, for every sampled parent in each iteration, **fires all registered mutators concurrently**, collecting their offspring into a single shared pool. The evolver has no knowledge of what a mutator does internally; it only knows that calling `mutate()` produces zero, one, or more child organisms.

This means a single-parent LLM rewriter, a multi-parent crossover synthesizer, a gradient-free optimizer, or a random-perturbation engine can all live in the same `mutators` list and operate simultaneously on the same population — with no special routing, no adapter layer, and no conditional dispatch. The independence is total: each mutator sees only the parent organism, the batch of failure cases, and the learning log entries handed to it. It is completely unaware of what other mutators exist.

***

## 2. Code Evidence: The `Mutator` ABC as the Atomic Interface

### 2.1 The Abstract Contract — `problem.py` L177–211

[`problem.py L177–211`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problem.py#L177-L211)

```python
class Mutator(ABC, Generic[OrganismT, EvaluationFailureCaseT]):
    @abstractmethod
    def mutate(
        self,
        organism: OrganismT,
        failure_cases: list[EvaluationFailureCaseT],
        learning_log_entries: list[LearningLogEntry],
    ) -> list[OrganismT]:
        raise NotImplementedError("Mutators must implement the mutate method")

    @property
    def supports_batch_mutation(self) -> bool:
        return False
```

**Why this is critical:** The entire interface is **three inputs and one output list**. The `@abstractmethod` decorator enforces that any subclass — regardless of its internal mechanism (LLM call, genetic crossover, MCMC perturbation, gradient-free optimization) — exposes exactly this surface. The `Generic[OrganismT, EvaluationFailureCaseT]` typing ensures type safety across all problem domains without coupling the interface to any specific domain.

***

### 2.2 The `Problem` Pydantic Model — `problem.py` L239–244

[`problem.py L239–244`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problem.py#L239-L244)

```python
class Problem(BaseModel, Generic[OrganismT, EvaluationResultT, EvaluationFailureCaseT]):
    initial_organism: OrganismT
    evaluator: Evaluator[OrganismT, EvaluationResultT, EvaluationFailureCaseT]
    mutators: list[Mutator[OrganismT, EvaluationFailureCaseT]]
```

**Why this is critical:** The composition point is a plain Python list of `Mutator` instances, not an enum, a factory, or a strategy map. New mutators are added by appending to this list. No registration, no switch-case dispatch, no evolver recompilation. The Pydantic model validates that all mutators share the same `OrganismT` and `EvaluationFailureCaseT` but imposes zero constraint on what they do internally.

***

### 2.3 The Concurrent Dispatch Loop — `evolver.py` L138–157

[`evolver.py L138–157`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L138-L157)

```python
for organism, evaluation_result in parents:
    failure_cases = evaluation_result.sample_trainable_failure_cases(batch_size=self._batch_size)
    learning_log_entries = self._learning_log_view.get_entries_for_organism(organism)
    for mutator in self._mutators:
        failure_cases_for_mutator = (
            failure_cases if mutator.supports_batch_mutation else [failure_cases[0]]
        )
        mutator_future = mutator_executor.submit(
            self._mutate_and_inject_attributes,
            organism,
            mutator,
            failure_cases_for_mutator,
            learning_log_entries,
        )
        mutator_futures.append(mutator_future)
```

**Why this is critical:** This double loop (outer: parents; inner: mutators) is the heart of heterogeneous dispatch. Every mutator in the list is submitted as an independent `Future` to a `ThreadPoolExecutor` (or `ProcessPoolExecutor`). The only per-mutator branch is the `supports_batch_mutation` check, which governs how many failure cases are forwarded — this is the **only** interface through which the evolver adapts its behavior to a specific mutator's capability. Everything else is uniform.

***

### 2.4 Attribute Injection After Dispatch — `evolver.py` L204–222

[`evolver.py L204–222`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L204-L222)

```python
def _mutate_and_inject_attributes(
    self,
    parent_organism: OrganismT,
    mutator: Mutator[OrganismT, EvaluationFailureCaseT],
    failure_cases: list[EvaluationFailureCaseT],
    learning_log_entries: list[LearningLogEntry],
) -> list[OrganismT]:
    mutated_organisms = mutator.mutate(parent_organism, failure_cases, learning_log_entries)
    cast_failure_cases: list[EvaluationFailureCase] = [c for c in failure_cases]
    for organism in mutated_organisms:
        if organism.parent is None:
            organism.parent = parent_organism
        if organism.from_failure_cases is None:
            organism.from_failure_cases = cast_failure_cases
    return mutated_organisms
```

**Why this is critical:** The evolver guarantees lineage bookkeeping uniformly for all mutators. A mutator that doesn't bother setting `organism.parent` still gets correct provenance tracked — the wrapper fills in defaults. This means even a naive mutator that returns a raw organism gets full lineage tracing for free, further reducing the implementation burden on new mutators.

***

## 3. Design Reasoning: Why Evolutionary Systems Require Pluggable, Independent Mutators

### 3.1 The "No Free Lunch" Theorem as Architectural Constraint

The No Free Lunch theorem for optimization states that no single search strategy outperforms all others across all problems. For a framework intended as a universal optimizer — one that operates on code, prompts, geometric arrangements, and arbitrary LLM-evaluable domains — this is not merely an academic concern: it directly implies that a single mutation strategy will be locally optimal for some problem classes and catastrophically suboptimal for others.

By making mutators independent and pluggable, the framework lets users compose strategies appropriate for their landscape: a gradient-rich problem may benefit from a focused single-failure-case repair mutator, while a deceptive multimodal problem benefits from crossover between high-scoring but structurally dissimilar parents. Neither strategy needs to know the other exists.

### 3.2 LLM Context Window as a Hard Engineering Constraint

The `supports_batch_mutation` flag is not a performance nicety — it encodes a fundamental constraint of LLM-based mutation. Providing multiple failure cases in a single prompt (batch mutation) consumes more context tokens. For some mutators (e.g., crossover, which already formats 2–3 parent code blocks plus the problem statement into the prompt), the budget is exhausted before additional failure cases can be included.

By exposing `supports_batch_mutation` as a per-mutator property, the evolver can query each mutator's own capacity declaration and adapt accordingly, without the evolver itself needing to know anything about token counts, prompt templates, or LLM APIs:

[`evolver.py L138–142`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L138-L142)

```python
failure_cases_for_mutator = (
    failure_cases if mutator.supports_batch_mutation else [failure_cases[0]]
)
```

### 3.3 Concurrent Futures: Thread Pool Safety as a Design Boundary

Each `mutator_executor.submit()` call dispatches a mutator into an independent thread. This is only safe because mutators share zero mutable state — they receive immutable `Organism` and `EvaluationFailureCase` objects, and they return new organism instances. The `MutatorContext` provides read-only population access via `self._context.population`. The atomic batch-commit pattern (collecting all futures before adding to population) ensures in-iteration offspring are invisible to co-running mutators:

```python
# Collect all evaluation results before we add them to the population.
# This makes sure that population updates are made atomically...
concurrent.futures.wait([evaluation_future for _, evaluation_future in organism_evaluation_futures])
for mutated_organism, evaluation_future in organism_evaluation_futures:
    evaluation_result = evaluation_future.result()
    self._population.add(mutated_organism, evaluation_result)
```

This atomicity guarantee is what makes heterogeneous concurrent mutators safe: a crossover mutator sampling from the population mid-iteration always sees the population from the *previous* iteration, preventing feedback loops between co-running mutators.

***

## 4. Mutators with Fundamentally Different Mechanisms: Code Evidence

The following four mutators co-exist in the ARC-AGI problem configuration and demonstrate maximum mechanism diversity:

### 4.1 `ArcAgiMutator` — Single-Parent Diagnostic Repair

[`arc_agi.py L788`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/arc_agi.py#L788)

```python
class ArcAgiMutator(Mutator[ArcAgiOrganism, ArcAgiEvaluationFailureCase]):
```

[`arc_agi.py L1025–1026`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/arc_agi.py#L1025-L1026)

```python
@property
def supports_batch_mutation(self) -> bool:
    return True
```

**Mechanism:** Given one parent and a batch of failure cases, this mutator constructs a prompt containing the full ARC problem statement, the current code, and per-input-output failure feedback. It then submits the prompt to an LLM (Gemini, Claude, or GPT depending on `USE_PROVIDER`) requesting a repaired `transform()` function. The operator works in the *code-repair* regime: it assumes the parent's structure is approximately correct and seeks targeted fixes. An `aggressive_fraction` parameter (default 0.5) randomly selects between a conservative "try to fix" prompt and a radical "start over" prompt.

***

### 4.2 `ArcAgiCrossoverMutator` — Multi-Parent Genetic Synthesis

[`arc_agi.py L1050`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/arc_agi.py#L1050)

```python
class ArcAgiCrossoverMutator(Mutator[ArcAgiOrganism, ArcAgiEvaluationFailureCase]):
    """Crossover mutator that combines explanations from multiple parent organisms."""
```

[`arc_agi.py L1138–1148`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/arc_agi.py#L1138-L1148)

```python
def mutate(self, organism, failure_cases, learning_log_entries, retries_remaining=1):
    if random.random() > self._crossover_frequency:
        return []
    ...
    parents_with_results = self._context.population.sample_parents(
        self._num_parents_per_crossover,
        replace=False,
        novelty_weight=self.SAMPLING_NOVELTY_WEIGHT,
    )
```

**Mechanism:** The crossover mutator executes on a probabilistic gate (`crossover_frequency=0.25` by default). When triggered, it accesses the population via `MutatorContext`, independently samples **three** diverse parents (with elevated `novelty_weight=1.0` to ensure structural diversity), and constructs a synthesis prompt containing all three parents' code and explanations. The LLM is asked to identify insights common across parents and divergences between them, then write a unified child. This is a fundamentally different *information source*: repair mutators consume failure cases, crossover mutators consume the population graph.

***

### 4.3 `ImproveParrotMutator` — Minimal Single-Case Prompt Evolution

[`parrot.py L~55`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/parrot.py)

```python
class ImproveParrotMutator(Mutator[ParrotOrganism, ParrotEvaluationFailureCase]):
    def mutate(self, organism, failure_cases, learning_log_entries) -> list[ParrotOrganism]:
        failure_case = failure_cases[0]
        prompt = jinja2.Template(self.IMPROVEMENT_PROMPT_TEMPLATE.strip()).render(
            organism=organism, failure_case=failure_case
        )
        improvement_response = _prompt_llm(prompt)
        improved_prompt_template = self._parse_response(improvement_response)
        return [ParrotOrganism(prompt_template=improved_prompt_template)]
```

**Mechanism:** This mutator evolves *prompt text*, not executable code. The organism is a Jinja2 template string. The failure case carries the phrase that wasn't repeated and the LLM's incorrect response. The mutator asks a different LLM to diagnose the failure and rewrite the template. It deliberately ignores learning log entries (no `{% if learning_log_entries %}` block). `supports_batch_mutation` defaults to `False` — it only ever sees one failure case. This demonstrates the minimalist extreme of the interface.

***

### 4.4 `ImproveCirclePackingMutator` — Geometric Optimization Code Evolution

[`circle_packing.py L173`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/circle_packing.py#L173)

```python
class ImproveCirclePackingMutator(Mutator[CirclePackingOrganism, CirclePackingEvaluationFailureCase]):
    """Uses an LLM to mutate the circle packing code."""
```

**Mechanism:** The organism here is a Python function body that calls `scipy.optimize`. The failure case carries `sum_of_radii`, `error_message`, and `output` of the packing algorithm's execution. The mutator's prompt is domain-specialized with mathematical knowledge about circle packing (hexagonal arrangements, edge effects, scipy's SLSQP solver). Unlike `ArcAgiMutator` which deals with grid transformations, this mutator guides the LLM toward numerical optimization theory. It also makes use of the learning log via Jinja2 templating, unlike the Parrot mutator.

***

### 4.5 Comparison Table: Mechanistic Diversity Across Mutators

| Property | `ArcAgiMutator` | `ArcAgiCrossoverMutator` | `ImproveParrotMutator` | `ImproveCirclePackingMutator` |
|---|---|---|---|---|
| **Parent count** | 1 | 3 (sampled from population) | 1 | 1 |
| **Organism type** | Python code (`transform()`) | Python code (multi-parent) | Jinja2 prompt string | Python code (`construct_packing()`) |
| **Failure case signal** | Grid I/O diffs | Ignored (uses scores) | Phrase + LLM response | Sum-of-radii + error output |
| **Uses learning log** | Yes | No | No | Yes |
| **`supports_batch_mutation`** | `True` | `False` (implicit) | `False` | `False` |
| **Probabilistic gate** | `aggressive_fraction` for mode | `crossover_frequency=0.25` | None | None |
| **Domain knowledge in prompt** | ARC visual reasoning | ARC synthesis across parents | Echo/verbatim repetition | Circle packing + scipy |
| **Information source** | Failure cases + code | Population graph | Failure case only | Failure case + learning log |

***

## 5. Mathematical Reasoning: Selection Weights and the Sigmoid-Novelty Formula

Before mutators are called, the evolver must select which parents to feed them. The selection formula directly determines the evolutionary pressure under which heterogeneous mutators operate.

### 5.1 The Weight Formula

[`population.py L362–371`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L362-L371)

The weight for each organism is:

$$
w_i = \sigma(s_i) \times b_i
$$

where the sigmoid performance score is:

$$
\sigma(s_i) = \frac{1}{1 + e^{-\lambda (s_i - \mu)}}
$$

and the novelty bonus is:

$$
b_i = \frac{1}{1 + \eta \cdot c_i}
$$

with:
- $s_i$ = organism's fitness score (0–1)
- $\lambda$ = sharpness (default **10.0**)
- $\mu$ = midpoint score (default = **75th percentile** of current population)
- $\eta$ = novelty weight (default **1.0**)
- $c_i$ = number of children the organism has already produced

[`population.py L384–395`](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L384-L395)

```python
def _compute_sigmoid_performance(self, evaluation_result, midpoint_score) -> float:
    return 1 / (1 + math.exp(-self._sharpness * (evaluation_result.score - midpoint_score)))

def _compute_novelty_bonus(self, organism, novelty_weight) -> float:
    num_children = len(self._children[organism.id])
    return 1 / (1 + novelty_weight * num_children)
```

### 5.2 Concrete Numeric Example: How Parameters Shift Evolutionary Pressure

Suppose we have three organisms with scores $s = [0.60, 0.75, 0.90]$ and children $c = [0, 5, 0]$. The current 75th-percentile midpoint is **$\mu = 0.75$**.

**Scenario A — Default configuration ($\lambda=10, \eta=1.0$):**

$$
\sigma(0.60) = \frac{1}{1+e^{-10(0.60-0.75)}} = \frac{1}{1+e^{1.5}} \approx 0.182
$$
$$
\sigma(0.75) = \frac{1}{1+e^{0}} = 0.500, \quad b_2 = \frac{1}{1+1 \cdot 5} = 0.167
$$
$$
\sigma(0.90) = \frac{1}{1+e^{-1.5}} \approx 0.818, \quad b_3 = \frac{1}{1+0} = 1.0
$$

Weights: $[0.182, 0.083, 0.818]$, normalized probabilities: $[0.168, 0.077, 0.755]$

The high-scoring novel organism (0.90, 0 children) is selected 75.5% of the time.

**Scenario B — Sharpness increased ($\lambda=20, \eta=1.0$):**

$$
\sigma(0.60) = \frac{1}{1+e^{3.0}} \approx 0.047, \quad \sigma(0.75) = 0.500, \quad \sigma(0.90) \approx 0.953
$$

Weights: $[0.047, 0.083, 0.953]$, normalized: $[0.043, 0.076, 0.881]$

The best organism now dominates even more — selection becomes more **exploitative**.

**Scenario C — Low sharpness ($\lambda=2, \eta=0.1$):**

$$
\sigma(0.60) = \frac{1}{1+e^{0.3}} \approx 0.426, \quad \sigma(0.75) = 0.500, \quad \sigma(0.90) \approx 0.574
$$
$$
b_2 = \frac{1}{1+0.1 \cdot 5} = 0.667
$$

Weights: $[0.426, 0.333, 0.574]$, normalized: $[0.319, 0.250, 0.431]$

With low sharpness and low novelty penalty, the distribution is nearly flat — **maximum exploration**, making underperformers much more likely to be selected as parents for mutation.

**Key insight for heterogeneous mutators:** The `ArcAgiCrossoverMutator` overrides the novelty weight to `1.0` in its own parent-sampling call (regardless of the global `novelty_weight` setting), deliberately biasing crossover toward less-explored parents. This is a second, independent selection policy layered inside one mutator without touching the evolver.

***

## 6. Architectural Diagram: Mutator Independence Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Problem Definition                                │
│  Problem(                                                                │
│    initial_organism = ArcAgiOrganism(...),                               │
│    evaluator        = ArcAgiEvaluator(...),                              │
│    mutators         = [                                                  │
│                          ArcAgiMutator(train_in, train_out, test_in),   │
│                          ArcAgiCrossoverMutator(train_in, ...)           │
│                       ]  ← plain list, zero coupling                    │
│  )                                                                       │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │ evolve_iteration()
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Evolver.evolve_iteration()                            │
│                                                                          │
│  parents = population.sample_parents(k)   ← sigmoid × novelty selection │
│                                                                          │
│  for (organism, eval_result) in parents:                                │
│    failure_cases = eval_result.sample_trainable_failure_cases()          │
│    learning_log = learning_log_view.get_entries_for_organism(organism)   │
│                                                                          │
│    for mutator in self._mutators:  ← independent iteration              │
│      │                                                                   │
│      ├─ mutator.supports_batch_mutation?                                 │
│      │     Yes → pass all failure_cases                                  │
│      │     No  → pass only failure_cases                              │
│      │                                                                   │
│      └─ mutator_executor.submit(                                         │
│              _mutate_and_inject_attributes,                              │
│              organism, mutator, failure_cases, learning_log              │
│         )  ← independent Future, no shared mutable state                │
│                                                                          │
└──────┬──────────────────────────────────────────────────────────────────┘
       │ Futures complete concurrently
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          Mutators Execute in ThreadPoolExecutor (Independently)         │
│                                                                          │
│  ArcAgiMutator.mutate()          ArcAgiCrossoverMutator.mutate()        │
│  ┌──────────────────┐            ┌───────────────────────────────┐      │
│  │ Build repair     │            │ Probabilistic gate: 25%       │      │
│  │ prompt (failure  │            │ Sample 3 diverse parents from │      │
│  │ cases + code)    │            │ population graph (via context)│      │
│  │ → LLM call       │            │ → Build synthesis prompt      │      │
│  │ → parse code     │            │ → LLM call (HIGH thinking)   │      │
│  │ → ArcAgiOrganism │            │ → ArcAgiOrganism              │      │
│  └────────┬─────────┘            └──────────────┬────────────────┘      │
│           │                                     │                       │
│           └───────────────┬─────────────────────┘                       │
│                           ▼                                              │
│                    [child organisms]                                     │
└──────┬────────────────────────────────────────────────────────────────-─┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│              Optional Verification → Evaluation → Atomic Add             │
│                                                                          │
│  for child in all_children:                                              │
│    if should_verify: evaluator.verify_mutation(child)                    │
│    evaluation_result = evaluator.evaluate(child)                         │
│                                                                          │
│  concurrent.futures.wait(all_evaluation_futures)  ← atomic barrier      │
│  for child, result in zip(children, results):                            │
│    population.add(child, result)  ← batch commit                        │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
  Next iteration: updated population feeds back into sample_parents()
```

***

## 7. Alternative Approach: Monolithic Multi-Strategy Dispatcher

A conventional alternative to the pluggable Mutator ABC would be a single `Mutator` class with internal strategy selection — an enum of mutation modes dispatched via `if/elif` branching:

```python
# Anti-pattern: coupled monolithic mutator
class EvolutionStrategy(enum.Enum):
    REPAIR = "repair"
    CROSSOVER = "crossover"
    RANDOM_RESTART = "random_restart"

class MonolithicMutator:
    def mutate(self, organism, failure_cases, strategy: EvolutionStrategy):
        if strategy == EvolutionStrategy.REPAIR:
            return self._do_repair(organism, failure_cases)
        elif strategy == EvolutionStrategy.CROSSOVER:
            return self._do_crossover(organism, failure_cases)
        elif strategy == EvolutionStrategy.RANDOM_RESTART:
            return self._do_restart(organism)
```

This pattern is common in workflow engines and ML hyperparameter tuners. It centralizes all mutation logic and allows a scheduler to explicitly select strategies based on iteration number, score trajectory, or other global signals.

***

## 8. Why the Independent Mutator Design Is Architecturally Superior

The monolithic dispatcher creates three categories of coupling costs that the current design avoids:

**1. Extension cost is linear in the monolith, constant in the ABC.**
Adding a new strategy to a monolith requires modifying the strategy enum, adding an elif branch, and potentially refactoring parameter handling for strategies with different signatures. Adding a mutator in the current design means creating a new class that extends `Mutator` — the evolver and all existing mutators are unchanged.

**2. Concurrent execution is unsafe in a monolith with shared state.**
The `MonolithicMutator` would need to serialize access to internal state across threads or pass costly deep copies. The current design makes safety trivial: there is no shared mutable state because each concrete mutator is an independent object. The `MutatorContext.population` is read-only during an iteration (writes are deferred to the atomic batch-commit), so even population-reading mutators like `ArcAgiCrossoverMutator` are safe.

**3. Mixed organism types would require runtime type checks in a monolith.**
Because the `Mutator` is generic (`Generic[OrganismT, EvaluationFailureCaseT]`), type safety across domain-specific organism types (code, prompt templates, geometric configurations) is enforced by Python's type checker at definition time. A monolith would require runtime `isinstance()` guards or unchecked casts.

**Trade-off acknowledged:** The pluggable design gives up centralized strategy scheduling. There is no built-in mechanism to say "after iteration 10, stop using `ArcAgiMutator` and switch entirely to crossover." The `crossover_frequency` probability gate on `ArcAgiCrossoverMutator` is a workaround — it embeds partial scheduling logic inside the mutator itself (via `self._crossover_frequency`), which slightly violates separation of concerns. A future `MutatorScheduler` abstraction wrapping the `list[Mutator]` with iteration-based gating would complete the design.

***

## 9. System Design Insight: The Mutator ABC as an Evolutionary Open-Endedness Primitive

The mutator independence design is an instance of the **Open-Closed Principle** applied to evolutionary computation: the evolver is *closed for modification* (no changes needed to add strategies) but *open for extension* (any strategy expressible as `mutate() → list[Organism]` is a valid plugin). This is what allows the same framework to achieve 95.1% on ARC-AGI-2 with Gemini and 34% with open-weight Kimi K2.5 using entirely different mutation strategies configured at the problem level, without any framework changes.

The deeper insight is that the framework treats mutation strategy selection as a **problem-level concern**, not a framework-level concern. The scientist designing the ARC problem decides to pair a repair mutator with a crossover mutator. The scientist designing the Parrot problem decides a single minimal mutator suffices. The evolver is agnostic. This mirrors the architecture of biological evolution itself: the environment (evaluator) and the replication machinery (mutator) are decoupled, and new replication strategies can emerge without changing the laws of selection.

