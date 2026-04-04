# Q1 — Why `1/(1 + novelty_weight × num_children)` Solves the Pure-Fitness Exploration Bottleneck


## Q1 The Darwinian Evolver uses a weighted sampling parent selection combining sigmoid-scaled performance scores with a novelty bonus calculated as 1/(1 + novelty_weight * num_children). Why does penalizing frequently-used organisms through the 1/num_children term solve a specific exploration bottleneck that a pure fitness-based selection would create?




Penalizing frequently-used parents with the $1/(1 + \text{novelty\_weight} \cdot \text{num\_children})$ factor prevents the search from collapsing onto a few early "lucky" high-fitness organisms. It forces mutation budget to rotate through less-explored lineages while still giving genuinely strong parents elevated, but not unlimited, reproductive opportunities.

---

## Observation: Parent Weights Mix Sigmoid Fitness with Child-Count Penalty

Darwinian Evolver does **not** pick parents purely by fitness or score ranking. Instead, it maintains a persistent archive of organisms and, each iteration, assigns every *eligible* organism a **sampling weight** that is the product of:

1. A **sigmoid-scaled performance term** based on its evaluation score relative to a dynamic midpoint.
2. A **novelty bonus** that decays as the organism accumulates more children in the archive.

Formally, for organism $i$ with score $\alpha_i$ and child count $n_i$:

$$w_i = \sigma(\alpha_i) \times h_i$$

where:

$$\sigma(\alpha_i) = \frac{1}{1 + e^{-\lambda(\alpha_i - m)}}$$

is the sigmoid performance term with sharpness $\lambda$ and midpoint $m$, and:

$$h_i = \frac{1}{1 + \tau n_i}$$

is the novelty bonus with novelty weight $\tau$.

Parents are then sampled proportionally to $w_i$. The key observation is that **$h_i$ shrinks as $n_i$ grows**, so even very fit organisms are gradually down-weighted as they are exploited, and selection pressure rebalances toward less-explored lineages.

Without this history-dependent penalty, a pure fitness-based rule (sample strictly by $\sigma(\alpha_i)$) would cause a handful of early high-scoring organisms to monopolize almost all parent slots, turning the search into a narrow hill-climb around those lineages and starving the rest of the archive of mutations.

---

## Repo References: Where `num_children` Enters Weighted Parent Selection

### Repo Evidence: Parent Sampling Driven by Computed Weights

[`population.py` L332–369](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L332-L369)

```python
def sample_parents(
    self,
    k: int,
    iteration: int | None = None,
    replace: bool = True,
    novelty_weight: float | None = None,
    exclude_untrainable: bool = True,
) -> list[tuple[Organism, EvaluationResult]]:
    """Sample k parents from the population using weighted sampling."""
    # To be eligible for parent selection, an organism must:
    # * have failed in at least one trainable evaluation task
    # * be viable
    eligible_organisms = [
        (organism, evaluation_result)
        for organism, evaluation_result in self._organisms
        if evaluation_result.is_viable
        and (not exclude_untrainable or len(evaluation_result.trainable_failure_cases) > 0)
    ]
    if not eligible_organisms:
        raise RuntimeError("No eligible organisms for parent selection")

    if novelty_weight is None:
        novelty_weight = self._novelty_weight
    weights = self._compute_weights(eligible_organisms, novelty_weight)

    if replace:
        return random.choices(eligible_organisms, weights=weights, k=k)
    else:
        probabilities = np.array(weights) / sum(weights)
        indices = np.random.choice(len(eligible_organisms), size=k, replace=False, p=probabilities)
        return [eligible_organisms[i] for i in indices]
```

Here:
- Only *viable* organisms with trainable failure cases can be parents.
- Parent choice is entirely controlled by the `weights` vector.

---

### Repo Evidence: Weight = Sigmoid Performance × Novelty Bonus

[`population.py` L371–405](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L371-L405)

```python
def _compute_weights(
    self, eligible_organisms: list[tuple[Organism, EvaluationResult]], novelty_weight: float
) -> list[float]:
    """Implements weighting according to section "A.2 Parent Selection" from Zhang et al. 2025."""
    midpoint_score = self._compute_midpoint_score()
    weights = []
    for organism, evaluation_result in eligible_organisms:
        sigmoid_performance = self._compute_sigmoid_performance(evaluation_result, midpoint_score=midpoint_score)
        novelty_bonus = self._compute_novelty_bonus(organism, novelty_weight)
        weight = sigmoid_performance * novelty_bonus

        assert weight >= 0
        weights.append(weight)

    return weights
```

[`population.py` L407–414](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L407-L414)

```python
def _compute_sigmoid_performance(self, evaluation_result: EvaluationResult, midpoint_score: float) -> float:
    """Compute the sigmoid-scaled performance of an evaluation result."""
    sigmoid_performance = 1 / (1 + math.exp(-self._sharpness * (evaluation_result.score - midpoint_score)))
    return sigmoid_performance
```

[`population.py` L416–424](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L416-L424)

```python
def _compute_novelty_bonus(self, organism: Organism, novelty_weight: float) -> float:
    """
    Compute the novelty bonus based on the number of children.

    This assigns a bonus to organisms that haven't been explored as much,
    encouraging diversity in the population.
    """
    num_children = len(self._children[organism.id])
    novelty_bonus = 1 / (1 + novelty_weight * num_children)
    return novelty_bonus
```

If `novelty_weight = 0`, then `novelty_bonus = 1` always — the system collapses to pure sigmoid-scaled fitness selection with no exploration bonus whatsoever.

---

### Repo Evidence: Maintaining `num_children` via Parent–Child Lineage Tracking

[`population.py` L94–131](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L94-L131)

```python
def add(self, organism: Organism, evaluation_result: EvaluationResult) -> None:
    """Add a new organism and its evaluation result to the population."""
    self._organisms.append((organism, evaluation_result))
    self._organisms_by_id[organism.id] = (organism, evaluation_result)

    # Track parent-child relationships
    if organism.parent is not None:
        self._children[organism.parent.id].append(organism.id)

    # Update best organism
    ...
```

Every time a child organism is accepted:
- `_children[parent.id]` grows by one entry.
- The next call to `_compute_novelty_bonus` for that parent reads `len(self._children[organism.id])`, which is now larger — reducing its future sampling probability automatically.

---

## Mathematical Reasoning: How $1/(1 + \tau \cdot n_i)$ Breaks the Rich-Get-Richer Loop

### Formal Weight and Selection Probability

For each eligible organism $i$:

$$\sigma_i = \frac{1}{1 + e^{-\lambda(\alpha_i - m)}} \quad \text{(sigmoid performance)}$$

$$h_i = \frac{1}{1 + \tau n_i} \quad \text{(novelty bonus)}$$

$$w_i = \sigma_i \cdot h_i \quad \text{(sampling weight)}$$

$$P(i) = \frac{w_i}{\displaystyle\sum_j w_j} \quad \text{(parent selection probability)}$$

**Key properties:**

1. For fixed $\alpha_i$, $w_i$ is a **decreasing function** of $n_i$. Each additional child reduces $h_i$ and hence $P(i)$.
2. For fixed $n_i$, $w_i$ is an **increasing function** of $\alpha_i$. Fitness is still rewarded.
3. $w_i > 0$ for all finite $n_i$, so no organism is ever completely excluded from selection.

---

### Concrete Numeric Example: How Child Count Reallocates Mutation Budget

**Setup:** Three organisms A, B, C with:

| Parameter | Value |
|-----------|-------|
| Midpoint $m$ | 0.5 |
| Sharpness $\lambda$ | 10 |
| Novelty weight $\tau$ | 1 |

#### Initial Stage — No Child Penalty Yet ($n_i = 0$)

| Organism | Score $\alpha_i$ | $\sigma_i$ | $h_i$ | $w_i$ | $P(i)$ |
|----------|-----------------|------------|--------|--------|---------|
| A | 0.9 | $\frac{1}{1+e^{-4}} \approx 0.982$ | 1.000 | 0.982 | **55.8%** |
| B | 0.6 | $\frac{1}{1+e^{-1}} \approx 0.731$ | 1.000 | 0.731 | **41.5%** |
| C | 0.2 | $\frac{1}{1+e^{3}} \approx 0.047$  | 1.000 | 0.047 | **2.7%** |

Even at the very start, organism C — despite being present in the population — effectively has no reproductive chance under pure fitness selection.

---

#### Later Stage — A and B Have Been Over-Exploited

After many iterations: $n_A = 10$, $n_B = 4$, $n_C = 0$ (C still untouched).

**Novelty bonuses:**

$$h_A = \frac{1}{1 + 1 \cdot 10} = \frac{1}{11} \approx 0.091$$

$$h_B = \frac{1}{1 + 1 \cdot 4} = \frac{1}{5} = 0.200$$

$$h_C = \frac{1}{1 + 0} = 1.000$$

**Updated weights and probabilities:**

| Organism | $\sigma_i$ | $h_i$ | $w_i$ | $P(i)$ |
|----------|------------|--------|--------|---------|
| A | 0.982 | 0.091 | 0.089 | **31.6%** |
| B | 0.731 | 0.200 | 0.146 | **51.8%** |
| C | 0.047 | 1.000 | 0.047 | **16.7%** |

**What changed vs. pure fitness?**

- A's probability dropped from **55.8% → 31.6%** — its over-exploitation is penalized.
- C's probability jumped from **2.7% → 16.7%** — a **6× increase with zero change in score**.
- If $\tau = 0$ (pure fitness), A would still hold ~55.8% regardless of how many children it has produced.

This reallocation is the mechanism that prevents the system from spending its entire compute budget on small variations of lineage A while C — which might unlock a qualitatively different solution — never gets explored.

---

## Design Reasoning: Why Child-Count Penalty Solves This Specific Bottleneck

### The Bottleneck: Fitness-Only Selection in Noisy, Compute-Limited Search

Darwinian Evolver operates under three critical constraints:

1. **Expensive evaluations.** Each organism may require many LLM calls and code executions (e.g., ARC-AGI, circle packing). The total evaluation budget per run is tightly constrained.
2. **Noisy or partially observed fitness.** Early scores often come from small test sets or stochastic LLM behavior; one organism may appear "best so far" largely due to random variance.
3. **Open-ended search in a huge design space.** Mutators can introduce large structural changes (new algorithms, prompt strategies, decompositions), so the space of possible lineages is enormous.

Under **pure fitness-based selection**, these constraints create a specific failure mode:

- A slightly better (or lucky) organism early in the run gets a disproportionately high fitness weight.
- Parent selection concentrates almost all mutation budget on that single lineage and its descendants.
- If that lineage encodes a fundamentally flawed design choice, the search is trapped in a "dead end" — an evolutionary cul-de-sac where all descendants are minor variations of the same broken approach.
- Other lineages are technically present but selected so rarely they cannot accumulate the mutations needed to become competitive.

### Why $1/(1 + \tau \cdot n_i)$ Is the Right Mechanism

The child-count penalty has properties that directly target this failure mode:

1. **History-dependent exploitation cap.** Unlike a static fitness transform, $h_i$ depends on how much the system has *already invested* in organism $i$'s neighborhood. Once enough children have been spawned, the system implicitly says: *"We understand this region — further investment has diminishing returns."*

2. **Soft, not hard, limits.** $h_i$ never reaches zero. Exceptional lineages can produce many children, just at diminishing rates — both extremes (monopoly and sudden cutoff) are avoided.

3. **Lineage-local, not globally random.** The penalty deterministically shifts probability mass from overused to underused organisms. It is not noise injection — it is structured redistribution.

4. **Problem-agnostic and computationally free.** No behavior embedding, no distance metric, no archive search — just `len(self._children[organism.id])`, a value already tracked for learning logs and lineage visualization.

---

## Alternatives: Other Designs and Why Child-Count Penalty Was Chosen

| Alternative | Mechanism | Why Not Chosen Here |
|---|---|---|
| **Behavior-Space Novelty Search** | Reward parents whose offspring are behaviorally distant from archive | Requires a robust behavior embedding for arbitrary code/prompts — hard to define generally |
| **Hard Child Cap** ($n_i \le K$) | Strictly limit each organism to $K$ children | Brittle: hard to pick $K$ across problems; prematurely cuts off genuinely strong lineages |
| **Rank-Based / Tournament Selection** | Selection probability by rank, not raw score | Still purely fitness-based — a consistently top-ranked lineage still dominates; no history-of-use signal |
| **Epsilon-Greedy / Thompson Sampling** | Multi-armed bandit over parent organisms | Requires per-organism posteriors or exploration parameters; adds significant machinery |
| **$1/(1 + \tau n_i)$ Novelty Penalty** ✅ | Soft cap via child count | Cheap, problem-agnostic, directly addresses over-exploitation, already tracked for other uses |

---

## Summary

The $1/(1 + \tau \cdot n_i)$ penalty solves the exploration bottleneck of pure fitness-based selection by:

1. **Making exploitation self-limiting:** As a parent produces more children, its weight automatically decreases, preventing any single lineage from monopolizing the compute budget.
2. **Keeping exploration positive:** Weights never reach zero, so every organism always has some chance of being selected, maintaining the diversity of the archive.
3. **Being history-aware, not just score-aware:** Unlike any static fitness transform, this mechanism encodes *"how much have we already invested here?"* — the exact question that pure fitness selection never asks.
4. **Costing essentially nothing:** The penalty reads a `len()` of a list that is already maintained for learning logs and lineage visualisation, making it a free exploration mechanism in a system where evaluations are extremely expensive.
