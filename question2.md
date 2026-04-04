# Q2 — Atomic Population Update: Why All Evaluations Must Complete Before Integration


## Q2 Examine the atomic population update strategy where all evaluation results complete before integration. Why is this pattern critical for maintaining statistically valid search dynamics?


In Darwinian Evolver, all children generated in an iteration are **evaluated to completion first**, and **only then** are their `(organism, evaluation_result)` pairs integrated into the `Population` in a single batch. This creates an *atomic boundary* between the **parent selection + mutation + evaluation** phase and the **population update** phase. As a result, every mutation in iteration $t$ is sampled from a fixed, snapshot population $P_t$, and the statistics that drive future selection (scores, percentiles, learning logs, novelty counts) are computed on a coherent state $P_t$ before transitioning cleanly to $P_{t+1}$.

---

## Observation: All Evaluations Finish Before Any `population.add()` Is Called

The key pattern is a `concurrent.futures.wait()` barrier placed **before** any `population.add()` calls. All mutation and evaluation work is done concurrently and asynchronously, but the commit step is strictly gated behind this barrier. This means:

- No child from iteration $t$ is ever visible in the population during iteration $t$.
- Parent selection, percentile computation, learning log reads, and novelty accounting all see a **fixed, immutable snapshot** of the population for the entire duration of an iteration.
- The transition from $P_t$ to $P_{t+1}$ is a single atomic operation, not a rolling stream of partial updates.

Without this pattern, faster-evaluating organisms would appear mid-iteration, distorting selection statistics in a way that depends on wall-clock evaluation latency rather than algorithmic intent.

---

## Code Evidence: The Barrier `wait()` Enforces Atomicity Before `population.add`

### Evidence 1 — Barrier Before All `add()` Calls

[`evolver.py` L174–181](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L174-L181)

```python
# Collect all evaluation results before we add them to the population.
# This makes sure that population updates are made atomically and organisms from this iteration
# aren't visible to mutators within the same iteration (including their learning logs).
concurrent.futures.wait([evaluation_future for _, evaluation_future in organism_evaluation_futures])

for mutated_organism, evaluation_future in organism_evaluation_futures:
    evaluation_result = evaluation_future.result()
    self._population.add(mutated_organism, evaluation_result)
```

The comment directly explains the intent. The `wait()` call is a hard barrier — it blocks until *every* evaluation future has completed. Only then does the loop call `self._population.add(...)`, which is the single method that updates `_organisms`, `_organisms_by_id`, `_children`, and the learning log. No child from iteration $t$ can be sampled as a parent or appear in logs during the same iteration.

---

### Evidence 2 — Evaluation Is Asynchronous but Integration Is Deferred

[`evolver.py` L120–181](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L120-L181)

```python
executor_type = ProcessPoolExecutor if self._use_process_pool_executors else ThreadPoolExecutor
with (
    executor_type(max_workers=self._mutator_concurrency) as mutator_executor,
    executor_type(max_workers=self._evaluator_concurrency) as evaluator_executor,
):
    ...
    for future in concurrent.futures.as_completed(mutated_organisms_futures):
        organism, should_evaluate = future.result()
        if should_evaluate:
            evaluation_future = evaluator_executor.submit(self._evaluator.evaluate, organism)
            organism_evaluation_futures.append((organism, evaluation_future))

    # BARRIER — wait for all evaluations to finish
    concurrent.futures.wait([f for _, f in organism_evaluation_futures])

    # COMMIT — only now integrate into population
    for mutated_organism, evaluation_future in organism_evaluation_futures:
        evaluation_result = evaluation_future.result()
        self._population.add(mutated_organism, evaluation_result)
```

Mutations and evaluations are fully asynchronous via thread/process pools. Results are buffered in `organism_evaluation_futures`. Only after the explicit `wait()` are they integrated. This preserves maximum parallelism without allowing partial state to leak into parent selection or log computation.

---

### Evidence 3 — Selection Statistics Are Computed Over the Entire Stable `_organisms` List

[`population.py` L176–210](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L176-L210)

```python
def get_score_percentiles(self, percentiles: list[float] = DEFAULT_PERCENTILES) -> dict[float, float]:
    scores = [evaluation_result.score for _, evaluation_result in self._organisms]
    if not scores:
        return {percentile: 0.0 for percentile in percentiles}

    scores.sort()
    n = len(scores)
    score_percentiles = {}
    for percentile in percentiles:
        k = (n - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            score_percentiles[percentile] = scores[int(k)]
        else:
            d0 = scores[int(f)] * (c - k)
            d1 = scores[int(c)] * (k - f)
            score_percentiles[percentile] = d0 + d1

    return score_percentiles
```

The midpoint score $m_t$ used in sigmoid weighting is derived from the 75th percentile of `_organisms`. If children were integrated mid-iteration, different parent-selection calls would see different score distributions depending on which futures happened to finish, making the selection distribution depend on wall-clock latency rather than intended algorithmic state.

---

### Evidence 4 — Learning Log and Child Relationships Only Written Inside `add()`

[`population.py` L60–85](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L60-L85)

```python
def add(self, organism: Organism, evaluation_result: EvaluationResult) -> None:
    self._organisms.append((organism, evaluation_result))
    self._organisms_by_id[organism.id] = (organism, evaluation_result)
    self._add_to_learning_log(organism, evaluation_result)
    parent = organism.parent
    if parent is not None:
        self._children[parent.id].append(organism.id)
```

[`population.py` L222–246](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L222-L246)

```python
def _add_to_learning_log(self, organism: Organism, evaluation_result: EvaluationResult) -> None:
    attempted_change = organism.from_change_summary
    if attempted_change is None:
        return
    if organism.parent is not None:
        parent_result = self._organisms_by_id[organism.parent.id][1]
    else:
        parent_result = None
    observed_outcome = evaluation_result.format_observed_outcome(parent_result)
    entry = LearningLogEntry(attempted_change=attempted_change, observed_outcome=observed_outcome)
    self._learning_log.add_entry(organism.id, entry)
```

All side effects — `_organisms`, `_organisms_by_id`, `_children`, and `_learning_log` — are gated behind `add()`. Since `add()` is only called after the barrier, mutators in iteration $t$ can never read learning log entries for iteration-$t$ children, preserving strict causal ordering.

---

## Mathematical Reasoning: Non-Atomic Updates Corrupt the Search Distribution

### The Intended Generational Model

In Darwinian Evolver, each iteration is designed to follow a clean generational step:

$$P_{t+1} = \mathcal{F}(P_t, \varepsilon_t)$$

where $\mathcal{F}$ is the evolution operator and $\varepsilon_t$ is random noise from mutations and evaluations in iteration $t$. The key requirement is that $P_{t+1}$ depends **only** on $P_t$ and $\varepsilon_t$ — making the process Markovian.

If children are integrated mid-iteration, the actual state during sampling becomes:

$$P'_{t+\delta} = P_t \cup \{\text{children whose futures completed by time } \delta\}$$

and the Markov property is broken because $P'_{t+\delta}$ depends on wall-clock evaluation timing, not just on $P_t$.

### How Non-Atomic Updates Skew the Sigmoid Midpoint

The parent weight for organism $i$ is:

$$w_i = \sigma(\alpha_i;\, m_t) \cdot h_i$$

where:

$$\sigma(\alpha_i;\, m_t) = \frac{1}{1 + e^{-\lambda(\alpha_i - m_t)}}$$

and $m_t$ is the percentile-based midpoint score:

$$m_t = \text{Percentile}_{75}\bigl(\{\alpha_j : (j, \alpha_j) \in P_t\}\bigr)$$

Under atomic updates, $m_t$ is computed over the *complete* score set in $P_t$. Under streaming integration, the effective midpoint is:

$$m'_t = \text{Percentile}_{75}\bigl(\{\alpha_j \in P_t\} \cup \{\alpha_k^{\text{child}} : \text{child } k \text{ finished early}\}\bigr)$$

Since $m'_t \neq m_t$ in general, the sigmoid weight $\sigma(\alpha_i;\, m'_t)$ differs from $\sigma(\alpha_i;\, m_t)$ for **every** organism, corrupting the entire selection distribution.

---

## Concrete Numeric Example: Midpoint Drift With vs Without Atomic Update

**Setup:**

| Parameter | Value |
|-----------|-------|
| Current population scores $S_t$ | $\{1, 2, 3, 4\}$ |
| New children scores (all 4) | $\{5, 6, 7, 8\}$ |
| Percentile for midpoint | 75th |
| Sharpness $\lambda$ | 5 |

---

### Case A — Correct Atomic Behavior

All 4 children evaluated and integrated together:

$$S_{t+1} = \{1, 2, 3, 4, 5, 6, 7, 8\}, \quad n = 8$$

75th percentile index: $k = (8-1) \times 0.75 = 5.25$

Interpolation between index 5 (value 6) and index 6 (value 7):

$$m_{t+1} = 6 \times (6 - 5.25) + 7 \times (5.25 - 5) = 6 \times 0.75 + 7 \times 0.25 = 6.25$$

Sigmoid weights for two organisms with $\lambda = 5$:

$$\sigma_{\text{old}}(4) = \frac{1}{1 + e^{-5(4 - 6.25)}} = \frac{1}{1 + e^{11.25}} \approx 0.000013 \quad \text{(near zero)}$$

$$\sigma_{\text{new}}(7) = \frac{1}{1 + e^{-5(7 - 6.25)}} = \frac{1}{1 + e^{-3.75}} \approx 0.977 \quad \text{(dominant)}$$

The old organism is effectively dropped; the new high-scorer dominates. This is **correct and intended** behavior.

---

### Case B — Non-Atomic Streaming (Only Children with Scores 7, 8 Finish Early)

Score set seen during premature parent selection:

$$S'_{t+1} = \{1, 2, 3, 4, 7, 8\}, \quad n = 6$$

75th percentile index: $k = (6-1) \times 0.75 = 3.75$

Interpolation between index 3 (value 4) and index 4 (value 7):

$$m'_{t+1} = 4 \times (4 - 3.75) + 7 \times (3.75 - 3) = 4 \times 0.25 + 7 \times 0.75 = 6.25$$

In this example, $m_{t+1}$ and $m'_{t+1}$ happen to match. However, consider a case where **only the lowest new child (score 5) finishes early**:

$$S''_{t+1} = \{1, 2, 3, 4, 5\}, \quad n = 5$$

75th percentile: $k = (5-1) \times 0.75 = 3.0$, which is exactly index 3 (value 4):

$$m''_{t+1} = 4$$

Now the midpoint drops to 4, and sigmoid weights become:

$$\sigma_{\text{old}}(4) = \frac{1}{1 + e^{-5(4 - 4)}} = \frac{1}{1 + e^{0}} = 0.5$$

$$\sigma_{\text{new child}}(5) = \frac{1}{1 + e^{-5(5 - 4)}} = \frac{1}{1 + e^{-5}} \approx 0.993$$

The midpoint has *dropped* due to incomplete data, making the old organism appear competitive again with a 50% weight — even though the full score distribution should push its weight nearly to zero. **This is a direct corruption of the intended selection pressure.**

---

## Design Reasoning: Why Atomic Updates Are Critical for Statistically Valid Search

### 1. Preserving Markovian, Generation-Level Search Dynamics

An atomic update preserves the Markovian property:

$$P_{t+1} \sim \mathcal{F}(P_t)$$

Each iteration's parent selection sees a *complete* $P_t$, and $P_{t+1}$ is formed by adding *all* children together. This ensures that:

- Score percentiles and sigmoid midpoints are computed on a statistically representative sample.
- Novelty bonuses $h_i = 1/(1 + \tau n_i)$ account for all children added in the previous batch, not just those that happened to be written before a given sampling call.
- Experimental results are **reproducible** across different hardware and concurrency settings, because search dynamics are not entangled with evaluation latency.

### 2. Preventing Latency-Driven Selection Bias

Evaluation cost in this system is highly variable:

- Different problems (ARC-AGI, circle packing, parrot) have different runtimes.
- Longer reasoning chains or more complex code produce longer evaluations.

If integration were streaming, organisms with faster evaluations would be added to the population earlier in the same iteration and could appear as parents, creating a feedback loop where **"fast-to-evaluate" lineages are favored** over "slow-to-evaluate but higher quality" ones. Atomic updates ensure selection depends only on scores and novelty, not on evaluation latency.

### 3. Keeping Learning Logs Causally Clean

Learning logs record `(attempted_change, observed_outcome)` pairs and are fed back to mutators as structured experience. If logs for some children were written mid-iteration while others were not:

- Mutators later in the same iteration would see **partial, timing-dependent histories**.
- Causal attribution ("this mutation pattern tends to help") would be corrupted by evaluation ordering rather than reflecting true problem structure.

Atomic updates ensure all log entries from iteration $t$ become available **together** as a coherent batch of experience for iteration $t+1$.

### 4. Correctness of Novelty Accounting

The novelty bonus:

$$h_i = \frac{1}{1 + \tau \cdot n_i}$$

reads `num_children = len(self._children[organism.id])`. Since `_children` is only updated inside `add()`, which is only called after the barrier, the child count $n_i$ for any parent in $P_t$ reflects only children from iterations $0, 1, \ldots, t-1$. No intra-iteration self-influence is possible.

---

## Alternatives: Why Streaming or Lock-Based Updates Are Insufficient

| Approach | Mechanism | Problem |
|---|---|---|
| **Streaming integration** | `add()` called as soon as each future resolves | Selection statistics depend on evaluation completion order — breaks Markov property, introduces latency bias |
| **Fine-grained locks on `_organisms`** | Lock shared state, allow concurrent reads/writes | Readers still see partially updated state; statistical validity depends on timing, not algorithm |
| **Steady-state EA** (replace one at a time) | Each new child probabilistically replaces an existing member | Incompatible with percentile-based midpoint, batch novelty, and batch learning logs; requires redesign of all selection components |
| **Atomic batch commit** ✅ (chosen design) | `wait()` barrier then sequential `add()` calls | Preserves Markov property, eliminates latency bias, keeps logs causally clean, enables reproducibility |

---

## ASCII Sequence Diagram: One Evolution Iteration With Atomic Population Update


```text
Evolution Iteration t
=====================

                          ┌─────────────────────────────────────────────────────┐
                          │ Population P_t                                      │
                          │ (fixed snapshot: _organisms, _children, logs)       │
                          └───────────────────────┬─────────────────────────────┘
                                                  │
                                                  │ sample_parents() from P_t
                                                  ▼
                          ┌─────────────────────────────────────────────────────┐
                          │ Parent Selection (weighted)                         │
                          │ uses scores + novelty in P_t                        │
                          └───────────────────────┬─────────────────────────────┘
                                                  │
                                                  │ for each parent
                                                  ▼
                          ┌─────────────────────────────────────────────────────┐
                          │ Mutators (thread/process)                           │
                          │ mutate parents → children                           │
                          └───────────────────────┬─────────────────────────────┘
                                                  │
                                                  │ for each child that passes
                                                  │ verification
                                                  ▼
                          ┌─────────────────────────────────────────────────────┐
                          │ Evaluator (thread/process)                          │
                          │ evaluate(child) → score                             │
                          └───────────────────────┬─────────────────────────────┘
                                                  │
                                                  │ futures buffered in
                                                  │ organism_evaluation_futures
                                                  │
                                                  │ (P_t is still UNCHANGED here)
                                                  │ (mutators still see only P_t)
                                                  ▼
          ┌──────────────────────────────────────────────────────────────────────────────┐
          │ GLOBAL BARRIER: concurrent.futures.wait(all eval futures)                   │
          │ ──────────────────────────────────────────────────────────────────────────── │
          │ All evaluations must complete before any add() is called                    │
          └──────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         │ for each (child, future) in
                                         │ organism_evaluation_futures
                                         ▼
                          ┌─────────────────────────────────────────────────────┐
                          │ Commit to Population                                │
                          │ population.add(child, score)                        │
                          │                                                     │
                          │ - append to _organisms                              │
                          │ - update _organisms_by_id                           │
                          │ - update _children[parent]                          │
                          │ - append learning-log entry                         │
                          └───────────────────────┬─────────────────────────────┘
                                                  │
                                                  │ after ALL adds complete
                                                  ▼
                          ┌─────────────────────────────────────────────────────┐
                          │ Population P_{t+1}                                  │
                          │ (new coherent state for next iteration)             │
                          └─────────────────────────────────────────────────────┘

```
> **Key insight:** The single `concurrent.futures.wait()` barrier is the entire mechanism.
> Everything above it reads from and operates on $P_t$.
> Everything below it writes to $P_{t+1}$.
> There is no code path where a child from iteration $t$ can influence
> selection, logs, or novelty counts *within* the same iteration —
> giving the system statistically valid and reproducible search dynamics
> across all hardware and concurrency configurations.

---

## Summary

The atomic population update pattern ensures statistically valid search dynamics by:

1. **Preserving the Markov property** — $P_{t+1}$ depends only on $P_t$ and iteration-$t$ randomness, never on evaluation completion timing.
2. **Eliminating latency-driven selection bias** — selection probabilities depend on scores and novelty, never on which LLM calls happen to finish first.
3. **Maintaining causal integrity of learning logs** — mutators in iteration $t$ never see log entries for iteration-$t$ children, preserving clean cause-and-effect attribution.
4. **Keeping novelty accounting consistent** — child counts $n_i$ are only incremented after the barrier, so novelty bonuses $h_i = 1/(1 + \tau n_i)$ always reflect the correct historical state.
5. **Enabling reproducibility** — because search dynamics are decoupled from evaluation latency, the same algorithm produces statistically equivalent behavior regardless of hardware, thread scheduling, or network timing.
