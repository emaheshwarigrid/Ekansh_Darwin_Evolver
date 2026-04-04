# Q4 — Why Batching Failures Across Multiple Mutators Outperforms Sending All Failures to a Single Mutator


## Q4 Why would providing all failure cases to a single mutator be less effective than batching failures across multiple mutators?



## Observation: Failure-Case Routing Is Designed for Multi‑Mutator, Mini‑Batch Style Training

Darwinian Evolver is designed to work from a **sampled failure-case batch per parent** rather than dumping every available failure into one giant mutator prompt. In the actual loop, batch-capable mutators receive that sampled batch, while non-batch mutators receive a single representative failure from it. This still creates a mini‑batch style search process: each call works from focused failure evidence, proposes targeted changes, and the population then selects which proposed children actually improve general fitness.

Concentrating all failures into one LLM call reduces diversity of hypotheses, overloads the context window, and produces a single “monolithic” fix attempt instead of many independent exploratory mutations.

---

## Evidence: Per‑Mutator Failure Batches Rather Than a Single Global Failure Set

### Evidence 1 – Problems Expose Trainable Failure Cases That Can Be Sampled

The README describes how problems feed weighted, sampleable failure cases into the evolver, not a single immutable list to be handled by one mutator.

[README.md – "Weighted failure case sampling"](https://github.com/imbue-ai/darwinian_evolver/blob/main/README.md#weighted-failure-case-sampling)

> "To implement weighted failure case sampling for your problem: Categorize the types of failures that can occur during evaluation … then sample failure cases to present to the mutator."

This shows that:

- Failure cases are sampled, not always used en masse.
- The evolver works from a sampled subset of the failure pool instead of presenting the full failure list to mutators.

Small repo snippet:

```python
failure_cases = evaluation_result.sample_trainable_failure_cases(batch_size=self._batch_size)
for mutator in self._mutators:
    failure_cases_for_mutator = (
        failure_cases if mutator.supports_batch_mutation else [failure_cases[0]]
    )
```

This is the key routing logic in `evolver.py`: the parent first gets one sampled failure batch, and then each mutator receives either that batch or one representative case from it.

Small repo snippet:

```python
failure_type = random.choices(
    list(failure_type_frequencies.keys()),
    weights=list(failure_type_frequencies.values()),
    k=1,
)[0]
```

This comes from `EvaluationResult.sample_trainable_failure_cases(...)` in `problem.py` and shows that the system does not pass all failures by default; it first samples a failure type and then samples a focused batch from that type.

### Evidence 2 – Mutators Are Independent Consumers of Failure Batches

The research blog describing Darwinian Evolver’s architecture explains that the system supports batch mutations and multiple mutators.

[Imbue research blog – "Batch mutations"](https://imbue.com/research/2026-02-27-darwinian-evolver/#batch-mutations)

> "Rather than providing only a single failure case to the mutator, we support exposing and analyzing multiple failure points simultaneously.  
> This is roughly equivalent to the use of mini-batches in Stochastic Gradient Descent."

[Imbue ARC-AGI‑2 blog – "Mutation strategies"](https://imbue.com/research/2026-02-27-arc-agi-2-evolution/#mutation-strategies)

> "We also use crossover mutations to combine new discoveries from across the population. The crossover mutator runs 25% of the time and samples three parents from the population instead of just one …"

Taken together:

- There are multiple mutators (standard LLM code mutator, crossover mutator, etc.), each invoked repeatedly.
- Each mutator call receives the parent’s sampled failure context, with batch-capable mutators seeing the sampled batch and non-batch mutators seeing one focused case, giving multiple, independent mutation attempts per iteration.

Small repo snippet:

```python
@property
def supports_batch_mutation(self) -> bool:
    """
    If True, the `mutate` method can accept multiple failure cases at once.
    """
    return False
```

This default in `problem.py` is important for the design argument: focused single-failure mutation is the baseline, and batch mutation is an explicit capability that mutators opt into.

The alternative design in the question—"provide all failure cases to a single mutator"—would contradict this architecture and remove most of the benefits of mini‑batching and mutator diversity described in the research docs.

---

## Mathematical View: Why One Huge Batch to One Mutator Underperforms Distributed Mini-Batches

### 1. Each Mutator Call Is a Stochastic Optimizer on a Subset of Failures

Conceptually, you can think of a mutator call as producing a gradient-like update from the failure cases it sees. Let:

- $F = \{f_1, f_2, \dots, f_N\}$ be the set of all failure cases for a parent.
- $p_i(C)$ be the probability that a mutator call with context $C$ produces a child that fixes failure $f_i$.

If we give all failures to one mutator once, we get a single draw:

- Context $C_{\text{all}} = F$
- For each failure $f_i$, success probability $p_i(F)$
- Overall chance of fixing $f_i$ in this iteration is just $p_i(F)$

If we instead split into $K$ mini-batches for $K$ independent mutator calls, each with context $C_k \subseteq F$, then:

- For each call $k$, the chance it **fails** to fix $f_i$ is $1 - p_i(C_k)$.
- Assuming independence, the chance that **all** $K$ calls fail to fix $f_i$ is the product of those terms.
- Therefore the chance that **at least one** call fixes $f_i$ is:

$$
P(\text{fix } f_i) = 1 - \bigl(1 - p_i(C_1)\bigr)\bigl(1 - p_i(C_2)\bigr)\dotsm\bigl(1 - p_i(C_K)\bigr)
$$

This grows quickly as you add more calls, as long as each $p_i(C_k)$ is non-zero.

---

### 2. Concrete Numeric Example: Single Mutator vs Distributed Mini-Batches

Assume:

- $N = 20$ failure cases total.
- We will run up to $5$ LLM mutation calls this iteration.

#### Scenario A — All failures to one mutator (one huge context)

- Single mutator call sees all 20 failures: $C_{\text{all}} = F$.
- Due to context overload and conflicting constraints, suppose the chance it fixes any given failure is only:

$$
p_i(F) = 0.05 \quad (5\%)
$$

For a particular failure $f_7$:

$$
P(\text{fix } f_7 \text{ in this iteration}) = 0.05 \quad (5\%)
$$

#### Scenario B — 5 mini-batches across 5 mutation calls

Now split failures into 5 non-overlapping batches of 4 failures each, and route each batch to a different mutator call:

- Each batch $C_k$ has 4 failures.
- Because the context is smaller and more focused, suppose the chance of fixing any given failure in its batch rises to:

$$
p_i(C_k) = 0.15 \quad (15\%)
$$

For failure $f_7$ (which appears in exactly one batch, say $C_3$):

$$
P(\text{fix } f_7 \text{ in this iteration}) = 0.15 \quad (15\%)
$$

You have already tripled the per-iteration fix probability for that failure.

#### Scenario C — Overlapping batches (each failure seen by 2 mutators)

Now imagine designing the batches so that each failure appears in 2 different mutator calls (for example two mutators with different prompts), each still with $p_i(C_k) = 0.15$. Then:

- Probability that the **first** call fails to fix $f_i$ is $1 - 0.15 = 0.85$.
- Probability that the **second** call also fails is $0.85$.
- Probability that **both** calls fail is:

$$
0.85 \times 0.85 = 0.7225
$$

- Probability that at least one call fixes $f_i$ is:

$$
P(\text{fix } f_i) = 1 - 0.7225 = 0.2775 \quad (27.75\%)
$$

So a single failure now has about $27.75\%$ probability of being fixed in this iteration, compared to $5\%$ in Scenario A — more than $5\times$ better.

---

## Design Reasoning: Why Multi‑Mutator Failure Batching Is Architecturally Superior

### 1. Diversity of Hypotheses vs One Monolithic Rewrite

Each mutator embodies a different mutation strategy: a standard "refine the code" mutator, a crossover mutator that merges logic from multiple parents, and possibly problem‑specific mutators (e.g., restructuring prompts vs changing algorithmic code).

If you give all failure cases to a single mutator call, you get one hypothesis—one proposed patch—trying to satisfy every failure description simultaneously:

- The LLM tends to produce a conservative, averaged solution that may partially address many failures but fully fix none.
- Conflicting signals (different failure modes requiring different strategies) create a tangled objective, like trying to fit all data in one massive gradient step.

By distributing failures across multiple mutators and calls:

- Each call can specialize on a subset of failure modes, exploring more radical changes tailored to those patterns.
- Different mutators can experiment with orthogonal ideas (new algorithm vs small bugfix vs hybridizing two parents).
- The population then applies selection to keep only those children that genuinely improve evaluation scores on the broader scoring set.

### 2. Mini‑Batch Signal vs Full‑Batch Overfitting

The Imbue paper explicitly compares batch mutations to mini‑batches in SGD. If you provide all failure cases to one mutator:

- The LLM is encouraged to "fit" the entire training set in one giant step.
- This promotes overfitting to idiosyncratic failures instead of improving the underlying general algorithm.
- In practice, prompts become cluttered; the model may fix narrow edge cases while silently regressing others.

With smaller, diverse mini‑batches:

- Each mutation sees different slices of the failure distribution.
- The system gets a stochastic, lower‑variance estimate of what direction of change tends to help across many failures, because only those children that improve scores on a separate scoring set survive.

### 3. LLM Context Window and Cognitive Load

LLMs have finite context and limited "reasoning bandwidth". Dumping all failure cases (e.g., dozens of long traces or ARC grids) into one prompt:

- Risks truncation or aggressive summarization.
- Forces the model to juggle many constraints simultaneously, which empirically leads to shallow, surface‑level fixes.

Smaller batches:

- Allow more detailed descriptions per failure case within the same token budget.
- Let the mutator focus on understanding a single pattern deeply (for example, "the model fails to handle carry in multiplication") and propose a targeted algorithmic fix.
- Avoid instruction dilution: the behavioral signal of each failure case is clearer.

### 4. Fairness and Robustness Across Failure Modes

Failure cases are usually heterogeneous: some represent common, easy‑to‑fix bugs; others are rare, structurally different pathologies.

A single mutator with all failures tends to prioritize the most numerous or most salient failures in the prompt, leaving rare ones unaddressed.

By sampling failures and distributing them across mutators:

- Rare but important failure modes can still receive dedicated attention in their own mini‑batches.
- The evolution loop can track which batches led to large score improvements, using the learning log, and gradually allocate more effort to fruitful regions of failure space.

### 5. Parallelism and Compute Efficiency

Multiple mutator calls with different batches can run in parallel, exploiting multi‑core CPUs or multiple LLM endpoints, and allowing early‑finishing mutations to be evaluated and filtered via post‑mutation verification.[web:67]

A design that routes all failures into one mutator call serializes the critical path behind a single, expensive LLM completion and reduces opportunities for parallel exploration of different hypotheses.

---

## Architectural Flow: Failure Case Batching Across Multiple Mutators

```text
                  ┌─────────────────────────────────────────┐
                  │          Population / Evaluator         │
                  │   (scores, full failure case corpus F)  │
                  └─────────────────────┬───────────────────┘
                                        │
                                        │ 1. Select parent organism O
                                        ▼
                  ┌─────────────────────────────────────────┐
                  │   Failure Case Sampler for Parent O     │
                  │                                         │
                  │  -  Start from all failures F_O          │
                  │  -  Weight / categorize by type          │
                  │  -  Draw one sampled batch C ⊂ F_O       │
                  └─────────────────────┬───────────────────┘
                                        │
                                        │ 2. Route sampled context to mutators
                                        ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                     Parallel Mutator Invocations                 │
        │                                                                  │
        │  Mutator A  ← C       Mutator B  ← C[0]    Mutator C  ← C ...    │
        │  ─────────────────    ─────────────────    ─────────────────     │
        │  -  Batch-capable       -  Single-case        -  Different prompt │
        │  -  Propose child A₁   -  Propose child B₁   -  Propose child C₁ │
        │  -  Maybe more A₂,A₃   -  Maybe more B₂,B₃   -  ...              │
        └───────────────────────┬──────────────────────────────────────────┘
                                │
                                │ 3. Evaluate children on scoring set
                                ▼
                  ┌─────────────────────────────────────────┐
                  │         Post-Mutation Verification      │
                  │   (optional filter per child/batch)     │
                  └─────────────────────┬───────────────────┘
                                        │
                                        │ 4. Keep only children
                                        │    with improved scores
                                        ▼
                  ┌─────────────────────────────────────────┐
                  │     Atomic population.add() batch       │
                  │   -  Append successful children         │
                  │   -  Update _children[parent]           │
                  │   -  Log (batch, mutator, outcome)      │
                  └─────────────────────────────────────────┘
```
