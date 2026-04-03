# WeightedSamplingPopulation vs FixedTreePopulation: Architectural Audit

**Repository:** `imbue-ai/darwinian_evolver` | **Audited by:** Ekansh Maheshwari

---

## 1. Observation: Two Distinct Population Strategies

The `darwinian_evolver` framework separates *what to optimize* (problem, mutators, evaluator) from *how to allocate exploration effort* (the population strategy). This separation is the core design tension in evolutionary algorithms: should the system focus on the best-known solutions, or ensure diverse coverage of the search space?

The framework answers this by offering two concrete strategies:

- **`WeightedSamplingPopulation`** treats the population as a continuously growing **global archive**. At every iteration, it samples parents probabilistically — better-scoring and less-explored organisms get higher selection weight. This is adaptive: as the population evolves and scores shift, selection pressure adjusts automatically. It is designed for open-ended optimization where score is a meaningful, continuous signal.

- **`FixedTreePopulation`** treats the population as a **layered tree**. Every iteration, the system identifies the "current generation frontier" (deepest generation) and gives every organism there an equal, fixed number of child slots. Score plays no role in selection. This is deterministic: the branching factor is fully user-controlled via a repeating pattern.

Both strategies share the same `Evolver` pipeline (mutation, verification, evaluation, learning log). The only divergence is inside `sample_parents`. This makes the choice of strategy a **pure search policy decision**, independent of the rest of the problem definition.

---

## 2. Repo References: WeightedSamplingPopulation

### Strategy Selection Entrypoint

[evolve_problem_loop.py L122–L139](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolve_problem_loop.py#L122-L139)

```python
if self._fixed_children_per_generation is not None:
    population_cls = FixedTreePopulation
    specific_kwargs = {"fixed_children_per_generation": ...}
else:
    population_cls = WeightedSamplingPopulation
    specific_kwargs = {"sharpness": ..., "midpoint_score_percentile": ..., "novelty_weight": ...}
```

A single CLI flag is the only branch point. If `--fixed_children_per_generation` is absent, `WeightedSamplingPopulation` is instantiated with score-aware hyperparameters; otherwise the tree variant takes over. Everything downstream (mutators, evaluators, learning logs) is unaffected by this choice.

---

### Eligibility Filter

[population.py L335–L340](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L335-L340)

```python
eligible_organisms = [
    (organism, evaluation_result)
    for organism, evaluation_result in self._organisms
    if evaluation_result.is_viable
    and (not exclude_untrainable or len(evaluation_result.trainable_failure_cases) > 0)
]
```

Before any weight is computed, organisms that are non-viable or have no trainable failure cases are excluded entirely. This ensures mutators are never handed a "finished" or broken organism as a parent — only organisms that still have room to improve can reproduce.

---

### Weight Computation Core

[population.py L359–L373](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L359-L373)

```python
def _compute_weights(self, eligible_organisms, novelty_weight):
    midpoint_score = self._compute_midpoint_score()
    weights = []
    for organism, evaluation_result in eligible_organisms:
        sigmoid_performance = self._compute_sigmoid_performance(evaluation_result, midpoint_score)
        novelty_bonus = self._compute_novelty_bonus(organism, novelty_weight)
        weights.append(sigmoid_performance * novelty_bonus)
    return weights
```

Each organism's weight is the product of two independent signals — performance and novelty. Neither alone is sufficient: pure performance leads to convergence on early winners; pure novelty ignores what actually works. The product keeps both forces in play simultaneously.

---

### Sigmoid Performance

[population.py L384–L387](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L384-L387)

```python
sigmoid_performance = 1 / (1 + math.exp(-self._sharpness * (evaluation_result.score - midpoint_score)))
```

Rather than using raw scores directly, the sigmoid compresses any score into `(0, 1)`. The `sharpness` (λ) parameter controls how steeply higher scores are preferred. The `midpoint_score` (m) is computed dynamically from the current score percentile, keeping selection centred as the population improves.

---

### Novelty Bonus

[population.py L395–L397](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L395-L397)

```python
num_children = len(self._children[organism.id])
novelty_bonus = 1 / (1 + novelty_weight * num_children)
```

Every time an organism produces a child, its `num_children` count increases. This bonus decreases as an organism is repeatedly used, acting as a **soft usage budget** — no organism can dominate the parent pool indefinitely, even if it has the highest score.

---

## 3. Repo References: FixedTreePopulation

### Pattern-Driven Parent Selection

[population.py L463–L481](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L463-L481)

```python
num_children_per_parent = self._fixed_children_per_generation[
    iteration % len(self._fixed_children_per_generation)
]
current_generation = self._get_current_generation_frontier()
parents = []
for parent in current_generation:
    parents.extend([parent] * num_children_per_parent)
```

No score lookup, no sigmoid, no novelty bonus, no eligibility filter. Every organism on the frontier is repeated exactly `c_t` times. The pattern is cyclic via modulo, enabling schedules like "wide early exploration, narrow later refinement."

---

### Frontier and Generation Computation

[population.py L484–L500](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L484-L500)

```python
def _get_current_generation_frontier(self):
    max_gen = max(self._compute_generation(org) for org, _ in self._organisms)
    return [(org, result) for org, result in self._organisms
            if self._compute_generation(org) == max_gen]

@staticmethod
def _compute_generation(organism):
    gen = 0
    current = organism
    while current.parent is not None:
        gen += 1
        current = current.parent
    return gen
```

The frontier is **not** the best-scoring organisms — it is strictly the most recently created generation, enforcing strict layer-by-layer expansion regardless of score.

---

### Pattern Wrap-Around (Test Confirmation)

[population_test.py L788–L790](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population_test.py#L788-L790)

```python
# Iteration 3: pattern [3, 2, 4] wraps around (3 % 3 = 0), uses 3 children
parents_iter3 = population.sample_parents(k=0, iteration=3)
assert len(parents_iter3) == 9  # 3 organisms × 3 children
```

The `k` argument is **ignored** in `FixedTreePopulation` — output size is fully determined by frontier size × pattern value.

---

## 4. Mathematical Reasoning

### WeightedSamplingPopulation Formula

For organism **i** with score **α_i**, child count **n_i**, midpoint **m**, sharpness **λ**, novelty weight **τ**:

$$w_i = \underbrace{\frac{1}{1 + e^{-\lambda(\alpha_i - m)}}}_{\text{sigmoid performance}} \;\times\; \underbrace{\frac{1}{1 + \tau \cdot n_i}}_{\text{novelty bonus}}$$

**Sigmoid performance** maps the raw score into (0, 1). At λ = 10, a score just 0.3 above the midpoint yields σ ≈ 0.95, while one 0.3 below yields σ ≈ 0.05 — near-binary split. At λ = 2, those same scores yield 0.73 and 0.27 — a much softer preference.

**Novelty bonus** halves an organism's effective weight each time its child count increases by 1/τ. At τ = 1, an organism with 5 children has novelty = 1/6 ≈ 0.167. At τ = 0, the bonus is always 1 — pure fitness selection.

**Midpoint m** is computed dynamically as a score percentile (default: 75th), keeping the sigmoid centred in the active score range as the population improves.

---

### Concrete Numeric Example

**Setup:** Three organisms, λ = 10, m = 0.5, τ = 1

| Organism | Score α | Children n | Sigmoid σ | Novelty h | Weight w | P(selected) |
|----------|---------|------------|-----------|-----------|----------|-------------|
| A        | 0.9     | 1          | ≈ 0.982   | 0.500     | ≈ 0.491  | **68.1%**   |
| B        | 0.6     | 3          | ≈ 0.731   | 0.250     | ≈ 0.183  | **25.4%**   |
| C        | 0.2     | 0          | ≈ 0.047   | 1.000     | ≈ 0.047  | **6.5%**    |

**Tweaking λ:** If λ is lowered to 2 with the same scores:

$$\sigma_A \approx 0.73, \quad \sigma_B \approx 0.55, \quad \sigma_C \approx 0.27$$

The weight gap between A and C shrinks from **20×** to just **2.7×** — a much more exploratory regime.

---

### FixedTreePopulation Formula

With frontier size **F_t** and pattern value **c_t = pattern[t mod |pattern|]**:

$$|\text{parents at iteration } t| = F_t \times c_t$$

For a pattern `[5, 3, 2]` starting from a single root:

| Iteration | Frontier F_t | c_t | Mutations Launched |
|-----------|-------------|-----|--------------------|
| 0         | 1           | 5   | 5                  |
| 1         | 5           | 3   | 15                 |
| 2         | 15          | 2   | 30                 |
| 3 (wraps) | 30          | 5   | 150                |

---

## 5. Design Reasoning: When Each Strategy Is Optimal

### WeightedSamplingPopulation — Adaptive Open-Ended Optimization

**Optimal when:**
- You are optimizing under a **reliable, continuous scalar objective** such as test pass rate or a geometric quality score.
- Score differences are **meaningful signals** — better organisms genuinely occupy better regions of the search space.
- Evaluations are **expensive** — you cannot afford to waste compute on clearly inferior organisms.

---

### FixedTreePopulation — Structured, Controllable Exploration

**Optimal when:**
- The **evaluator is highly noisy or near-binary** (e.g., pass/fail, exact-match) making score-based weighting unreliable.
- You need **fair, deterministic expansion** for controlled ablations — every frontier organism must be explored equally.
- The problem is **simple or fast-converging** — systematic breadth-first exploration finds the solution quicker than adaptive focus on an early spurious score leader.
- You need to **predict and cap the total compute budget** before the run starts.

---

## 6. Problem-by-Problem Strategy Analysis

The `problems/` folder contains six files. The four runnable problems split evenly between the two strategies. The two non-runnable files (`arc_agi_poetiq.py` and `registry.py`) are analysed with **stated assumptions** about hypothetical standalone use.

| Problem File | What It Optimizes | Score Signal | Evaluator Noise | Recommended Strategy | Why This Strategy Wins |
|---|---|---|---|---|---|
| `parrot.py` | Prompt causing LLM to verbatim-repeat a phrase | Exact-match fraction — near-binary (0 = failed, 1 = matched) | **Low** — deterministic string match | **FixedTree** ✓ | When most organisms score 0, all sigmoid weights collapse to ~0 and selection becomes random. FixedTree pattern `[3, 2]` gives every prompt variant equal slots for fair comparison. WeightedSampling adds no information in a near-binary score regime. |
| `circle_packing.py` | Python code for geometric circle packing | Continuous sum of radii — **truly continuous, deterministic** | **None** — pure math, no randomness | **WeightedSampling** ✓ | Continuous, noise-free score makes sigmoid weighting highly meaningful. The 30-minute per-evaluation timeout makes it critical to focus compute only on high-scoring organisms. FixedTree would waste expensive slots on poor performers. |
| `multiplication_verifier.py` | Prompt to classify multiplication results | Accuracy % over a test set — near-binary per sample | **Medium** — LLM re-runs can yield different verdicts | **FixedTree** ✓ | Noisy LLM evaluator means an organism could score high on one run by luck. Weighted selection would over-promote that organism, monopolising future mutations. FixedTree prevents a single noisy high-scorer from crowding out alternatives. |
| `arc_agi.py` | Python code solver for ARC-AGI grid tasks | Multi-component composite: correctness + transfer + simplicity | **High** — multiple expensive LLM judges, each can vary | **WeightedSampling** ✓ (strongly) | Despite high noise, the composite score meaningfully distinguishes strong from weak solvers over many tasks. Adaptive allocation is essential when each evaluation costs significant API budget across three separate judge dimensions. |
| `arc_agi_poetiq.py` *(assumed)* | **Assumption:** If evolved as a standalone problem, this module would optimize the **soft scoring function and prompt templates** for ARC grid evaluation — i.e., evolve `soft_score()` to better correlate with human judgments of task correctness | Continuous correlation score between predicted grid similarity and ground-truth human ratings — **continuous, meaningful gradient** | **Low-Medium** — correlation is deterministic given fixed data, but the ground-truth dataset itself introduces variance | **WeightedSampling** *(assumed)* | **Assumption basis:** The `soft_score` function returns a float in [0, 1] via element-wise array comparison — a continuous, differentiable signal that improves gradually as the scoring formula is refined. This matches WeightedSampling's strength. FixedTree would be wasteful because the score gradient is real and informative even for small improvements in formula quality. |
| `registry.py` *(assumed)* | **Assumption:** If evolved as a standalone problem, the registry would optimize **problem routing and selection composition** — i.e., which problems to include, their weighting, and how to compose multi-problem evaluation pipelines | Binary per-problem inclusion/exclusion — a discrete combinatorial signal (problem A works in the pipeline or it doesn't) | **High** — meta-level evaluation across multiple problem types introduces compound noise from each sub-evaluator | **FixedTree** *(assumed)* | **Assumption basis:** Evolving a registry is fundamentally a combinatorial search over a discrete set of options (include/exclude problems, reorder them). With binary inclusion decisions and compounded evaluator noise, sigmoid weighting on a composite meta-score would be unreliable. FixedTree's equal branching systematically covers the combinatorial space depth-first without premature commitment to a noisy early winner. |

**Key pattern:** FixedTree wins when the score is near-binary or evaluator noise is high enough to make weighting misleading. WeightedSampling wins when the score is continuous, reliable, and expensive-to-compute — conditions where adaptive allocation has genuine leverage.

---

## 7. Alternative Approaches Considered

| Alternative | Why Rejected |
|---|---|
| Single `Population` class with a `mode` flag | Mutually exclusive fields accumulate; `sample_parents` grows complex conditionals; snapshot logic becomes fragile |
| Emulate fixed tree via extreme hyperparameters (λ → 0, τ → ∞) | Still archive-style; no hard guarantees on per-frontier-node expansion counts |
| Push selection logic into mutators/evaluators | Breaks responsibility separation; population selection is the correct interface for "who gets to reproduce" |

---

## 8. Architectural Flow Diagram

The ASCII diagram below renders in any viewer. The Mermaid source is included beneath it for use in GitHub, Obsidian, or any Mermaid-compatible renderer.

### Rendered (ASCII — universal)

```
                 ┌────────────────────────────────────────┐
                 │          EvolveProblemLoop              │
                 │  --fixed_children_per_generation ?      │
                 └──────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │ PROVIDED?                                   │
              ▼ YES                                         ▼ NO
  ┌───────────────────────┐                ┌──────────────────────────────┐
  │  FixedTreePopulation  │                │  WeightedSamplingPopulation  │
  │  [pattern list]       │                │  sharpness + midpoint        │
  │                       │                │  + novelty_weight            │
  └──────────┬────────────┘                └─────────────┬────────────────┘
             │                                           │
             ▼                                           ▼
  ┌─────────────────────────┐         ┌──────────────────────────────────┐
  │   sample_parents()      │         │        sample_parents()          │
  │                         │         │                                  │
  │ 1. _get_frontier()      │         │ 1. Filter: viable +              │
  │    walk .parent links   │         │    has trainable failures        │
  │    → find max gen       │         │                                  │
  │                         │         │ 2. _compute_midpoint_score()     │
  │ 2. Pattern lookup:      │         │    (percentile of pop scores)    │
  │    pattern[iter % len]  │         │                                  │
  │    → c_t children each  │         │ 3. sigmoid(λ × (score - m))     │
  │                         │         │    → performance weight          │
  │ 3. Repeat each frontier │         │                                  │
  │    org c_t times        │         │ 4. 1 / (1 + τ × num_children)   │
  │                         │         │    → novelty bonus               │
  │ ⚠ SCORE IGNORED         │         │                                  │
  │   all get equal slots   │         │ 5. random.choices(weights)       │
  └──────────┬──────────────┘         │    → k parents sampled           │
             │                         └─────────────┬────────────────────┘
             │                                       │
             └─────────────────┬─────────────────────┘
                               ▼
              ┌────────────────────────────────────┐
              │      Evolver.evolve_iteration       │
              │   Mutate  →  Verify  →  Evaluate    │
              └────────────────────┬───────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │          population.add()           │
              │  _children[parent.id]               │
              │    .append(child.id)                │
              │  (drives novelty bonus next iter)   │
              └────────────────────┬───────────────┘
                                   │
                                   ▼
                            Next iteration
```



## 9. System Design Insight

Both strategies share the same `Evolver` pipeline (mutators, evaluators, learning logs) and the same `Population` base class. The only divergence is in `sample_parents`. This means:

- You can **swap strategies on resume** by changing only the `--fixed_children_per_generation` flag.
- Your entire problem definition (mutators, evaluator, batch size) is **fully portable** between regimes.
- `WeightedSamplingPopulation` is the default because most real-world optimization problems benefit from adaptive, score-guided allocation. `FixedTreePopulation` is an explicit opt-in for problems where determinism, fairness, or near-binary scoring makes weighted selection unreliable or wasteful.
"""
