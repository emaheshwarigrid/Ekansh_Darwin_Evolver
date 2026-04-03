# Replicator Dynamics and Genetic Lineages in `imbue-ai/darwinian_evolver`

> **Architectural Audit Question:** Explain the replicator dynamics of the population and how genetic lineages structure problem-solving. Why does tracking parentage (via `_children defaultdict`) enable specific optimization strategies?

***

## 1. Observation: How the Evolutionary Replicator Loop is Implemented

The `darwinian_evolver` repository implements a full **replicator-selector loop** directly analogous to Darwinian natural selection — but operating over LLM-generated Python programs (organisms). Each organism is a solution candidate (e.g., an ARC-AGI grid-transform function). The population grows monotonically: organisms are never removed; they are only added. Selection pressure is expressed not by deletion but by the *probability of being chosen as a parent*.

The core framework runs for the configured number of iterations, while some experiment scripts add task-specific stopping criteria on top. At each iteration:

1. **Selection** — `sample_parents()` draws `k` organisms from the population, weighted by a sigmoid-fitness × novelty function.
2. **Mutation** — Each selected parent is mutated (via LLM) to produce one or more offspring.
3. **Evaluation** — Offspring are scored against the task's training cases.
4. **Insertion** — Viable offspring are permanently added to the archive.
5. **Lineage bookkeeping** — `_children[parent.id].append(child.id)` registers the parent–child edge in the organism graph.

This constitutes a discrete-generation **replicator dynamic**: organisms with higher relative fitness replicate into more offspring, compounding advantage across the search tree.

***

## 2. Code Evidence: The `_children` Defaultdict as the Lineage Spine

### Declaration — The Directed Parentage Graph

[`population.py` L26–L29](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L26-L29)
```python
_organisms: list[tuple[Organism, EvaluationResult]]
_organisms_by_id: dict[UUID, tuple[Organism, EvaluationResult]]
_children: defaultdict[UUID, list[UUID]]
_learning_log: LearningLog
```
`_children` is typed as `defaultdict[UUID, list[UUID]]` — a **directed adjacency list** keyed on parent UUID, mapping to an ordered list of child UUIDs. Using `defaultdict(list)` means any UUID access on a node with zero offspring silently returns `[]` without a `KeyError`, which is critical because *most organisms will never become parents* (they are evolutionary dead-ends). This choice removes an entire class of `KeyError` guards throughout the codebase.

### Initialization and Seeding

[`population.py` L44–L46](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L44-L46)
```python
self._organisms = []
self._organisms_by_id = {}
self._children = defaultdict(list)
```
The constructor initializes `_children` as empty. The root organism has no parent (`parent is None` assertion at L38), so it is never registered as a child. It begins as the sole node in the graph.

### The `add()` Write Path — How Edges Are Created

[`population.py` L130–L138](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L130-L138)
```python
def add(self, organism: Organism, evaluation_result: EvaluationResult) -> None:
    self._organisms.append((organism, evaluation_result))
    self._organisms_by_id[organism.id] = (organism, evaluation_result)
    self._add_to_learning_log(organism, evaluation_result)
    parent = organism.parent
    if parent is not None:
        self._children[parent.id].append(organism.id)
```
Every time a new organism enters the population, if it has a parent, its UUID is appended to `_children[parent.id]`. This is an **O(1) amortized** edge insertion. The call is a direct side-effect of `add()`, so it is *impossible* to add an organism without registering the lineage edge. This enforces structural integrity of the organism DAG.

### Snapshot Reconstruction — Lineage Replay

[`population.py` L66–L74](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L66-L74)
```python
population._children = defaultdict(list)
population._learning_log = LearningLog()
population._organisms_failed_verification = snapshot_dict.get(...)
for organism, evaluation_result in population._organisms:
    population._add_to_learning_log(organism, evaluation_result)
    parent = organism.parent
    if parent is not None:
        population._children[parent.id].append(organism.id)
```
During `from_snapshot()`, `_children` is *not* serialized directly — it is **reconstructed from the parent pointer chain**. This is a deliberate design: the `parent` reference embedded in each `Organism` Pydantic model is the single source of truth. `_children` is a derived, in-memory index (an inverted view). The snapshot only needs to preserve the objects; the graph topology is always recomputable.

### Read Path — `get_children()` for Graph Traversal

[`population.py` L213–L215](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L213-L215)
```python
def get_children(self, parent: Organism) -> list[tuple[Organism, EvaluationResult]]:
    return [self._organisms_by_id[child_id] for child_id in self._children[parent.id]]
```
`get_children()` is the public API that converts UUID references back into full `(Organism, EvaluationResult)` tuples. It combines the `_children` index with the `_organisms_by_id` lookup, giving O(degree) traversal. This is used by `NeighborhoodLearningLogView` during graph-distance traversal (see Section 5).

***

## 3. Code Evidence: Parent Pointer in the Organism Model

[`problem.py` L43–L44](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problem.py#L43-L44)
```python
parent: Organism | None = None
additional_parents: list[Organism] = Field(default_factory=list)
```
Each `Organism` carries a **direct object reference** to its parent (not just a UUID). This enables upward traversal (ancestor walking) without any population lookup. `additional_parents` supports crossover — an organism can have up to 2–3 parents combined by the LLM mutator. The `from_snapshot()` method uses `pickle` specifically to *preserve these object identity relationships* (see snapshot comments at L83–L86 in `population.py`).

[`evolver.py` L213–L215](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L213-L215)
```python
for organism in mutated_organisms:
    if organism.parent is None:
        organism.parent = parent_organism
```
The `_mutate_and_inject_attributes` method guarantees that every offspring has its `parent` field set. If a mutator forgets to set it, the evolver injects it automatically. This is a **defensive lineage guarantee** — the parentage graph cannot have orphan nodes.

***

## 4. Mathematical Reasoning: Replicator Dynamics via Sigmoid-Weighted Selection

### The Weight Formula

The core selection weight for each organism is a product of two terms:

$$
w_i = \sigma_{\text{perf}}(s_i) \cdot \text{NovBonus}(n_i)
$$

where:

$$
\sigma_{\text{perf}}(s_i) = \frac{1}{1 + e^{-\alpha(s_i - m)}}
$$

$$
\text{NovBonus}(n_i) = \frac{1}{1 + \lambda \cdot n_i}
$$

- $s_i$ = score of organism $i$
- $m$ = midpoint score (dynamic, typically 75th percentile of current population)
- $\alpha$ = sharpness parameter (default: 10.0)
- $n_i$ = number of children of organism $i$ (read directly from `_children[organism.id]`)
- $\lambda$ = novelty weight (default: 1.0)

The **selection probability** for organism $i$ is:

$$
P_i = \frac{w_i}{\sum_{j} w_j}
$$

[`population.py` L384–L397](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L384-L397)
```python
def _compute_sigmoid_performance(self, evaluation_result, midpoint_score):
    return 1 / (1 + math.exp(-self._sharpness * (evaluation_result.score - midpoint_score)))

def _compute_novelty_bonus(self, organism, novelty_weight):
    num_children = len(self._children[organism.id])
    return 1 / (1 + novelty_weight * num_children)
```
The `_compute_novelty_bonus` function directly queries `_children` to get $n_i$. **Without `_children`, computing the novelty penalty would require a full linear scan of all organisms to count children for each candidate — O(N) per candidate, O(N²) per iteration.** The defaultdict gives O(1) lookup.

### Concrete Numeric Example: How Lineage Depth Kills Overexploited Parents

Consider a population of 4 organisms after 3 iterations, with `sharpness=10`, `midpoint=0.5`, `novelty_weight=1.0`:

| Organism | Score $s_i$ | Children $n_i$ | $\sigma_{\text{perf}}$ | NovBonus | Weight $w_i$ | $P_i$ |
|----------|----------------|-------------------|---------------------------|----------|----------------|---------|
| A (root) | 0.3 | 5 | $\frac{1}{1+e^{10(0.3-0.5)}} = 0.119$ | $\frac{1}{1+5} = 0.167$ | 0.0199 | **3.4%** |
| B | 0.6 | 1 | $\frac{1}{1+e^{10(0.6-0.5)}} = 0.731$ | $\frac{1}{1+1} = 0.500$ | 0.3655 | **62.5%** |
| C | 0.55 | 0 | $\frac{1}{1+e^{10(0.55-0.5)}} = 0.622$ | $\frac{1}{1+0} = 1.000$ | 0.6220 | **18.9%** |
| D | 0.7 | 2 | $\frac{1}{1+e^{10(0.7-0.5)}} = 0.881$ | $\frac{1}{1+2} = 0.333$ | 0.2937 | **15.2%** |

**Key observations:**

- Organism A (root, score=0.3, 5 children) has selection probability *3.4%* — nearly extinct despite being viable. Its novelty bonus has collapsed to 0.167 because it has been over-exploited.
- Organism C (score=0.55, 0 children) has a probability of 18.9% — almost as likely as D (score=0.7, 2 children) despite a lower score. The novelty bonus *doubles* its effective weight.
- If `novelty_weight=0`, D dominates at ~57%, starving out C. If `novelty_weight=2`, C's bonus becomes $1/(1+2*0)=1.0$ while D's becomes $1/(1+2*2)=0.2$, dramatically shifting toward unexplored branches.

**This is classic replicator dynamics**: the population tracks its own exploitation history via `_children` and self-regulates the exploration-exploitation trade-off without any external memory structure.

***

## 5. Code Evidence: `_children` Enabling the `NeighborhoodLearningLogView` Strategy

[`learning_log_view.py` L74–L91](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/learning_log_view.py#L74-L91)
```python
def _traverse_graph(self, organism, current_distance, visited):
    if current_distance > self._max_distance:
        return []
    visited.add(organism.id)
    organisms_in_range = [organism]
    if organism.parent is not None and organism.parent.id not in visited:
        organisms_in_range.extend(self._traverse_graph(organism.parent, current_distance + 1, visited))
    for child, _ in self._population.get_children(organism):
        if child.id not in visited:
            organisms_in_range.extend(self._traverse_graph(child, current_distance + 1, visited))
    return organisms_in_range
```
`NeighborhoodLearningLogView` performs a **bidirectional BFS** over the organism DAG — ascending through parents AND descending through children — up to a configured `max_distance`. This means when mutating organism B, the LLM receives learning log entries from B's parent, B's siblings (other children of the same parent), and B's own offspring. Without `_children`, only upward (ancestor) traversal would be possible. The downward traversal into sibling/offspring branches is **exclusively enabled by the `_children` index**.

[`learning_log_view.py` L30–L50](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/learning_log_view.py#L30-L50)
```python
class AncestorLearningLogView(LearningLogView):
    def get_entries_for_organism(self, organism):
        entries = []
        current_ancestor = organism
        while current_ancestor is not None and ...:
            maybe_entry = self._learning_log.get_entry(current_ancestor.id)
            if maybe_entry is not None:
                entries.append(maybe_entry)
            current_ancestor = current_ancestor.parent
        return entries
```
`AncestorLearningLogView` uses only the forward `parent` pointer — no `_children` needed — walking straight up the lineage chain. This gives the LLM a **causal history** of mutations that led to the current organism (like a git log). The `NeighborhoodLearningLogView` extends this by also seeing the "sibling experiments" that diverged from the same ancestor.

***

## 6. Code Evidence: `FixedTreePopulation` — Generation Frontier via `_children`

[`population.py` L484–L499](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/population.py#L484-L499)
```python
def _get_current_generation_frontier(self):
    if not self._organisms:
        return []
    max_gen = max(self._compute_generation(org) for org, _ in self._organisms)
    return [(org, result) for org, result in self._organisms if self._compute_generation(org) == max_gen]

@staticmethod
def _compute_generation(organism):
    gen = 0
    current = organism
    while current.parent is not None:
        gen += 1
        current = current.parent
    return gen
```
`FixedTreePopulation` walks upward through `parent` pointers to compute generation depth, then selects all organisms at `max_gen` as the frontier. Each frontier organism is then assigned exactly `fixed_children_per_generation[iteration % len(pattern)]` children. This enables **grid-search-style breadth-first expansion** (e.g., `[3, 2, 4]` means 3 children per parent in gen 1, 2 in gen 2, 4 in gen 3) rather than fitness-proportionate sampling. The important contrast is that `_children` is not used to form this frontier; the frontier logic is parent-pointer based.

***

## 7. Design Reasoning: Why `_children` Is the Right Data Structure for Evolutionary Optimization

### The Exploration-Exploitation Trade-off Requires Child-Count at Query Time

The novelty bonus formula must be evaluated *at every parent sampling call* for every eligible organism. If `_children` did not exist, computing $n_i$ would require a full O(N) scan of all organisms for each candidate, making `_compute_weights()` an O(N²) operation in the number of eligible organisms. With the defaultdict, child-count lookup is O(1), which is exactly the access pattern the novelty term needs.

### Graph Bidirectionality Is Needed for Context-Aware LLM Mutation

The `Organism` model stores a parent *object reference* (downward-navigable from root is impossible via this alone). `_children` is the **inverted index** that makes the DAG bidirectionally traversable. This directly enables the `NeighborhoodLearningLogView`, which feeds the LLM a rich neighborhood of *what happened in sibling branches* — not just the current lineage's history. The LLM can then distinguish between mutations that were tried and failed in adjacent branches vs. mutations that haven't been explored yet. This is analogous to the **learning log** feature described in Imbue's blog as a "differential signal about what actually moved the score".

### Snapshot Integrity Without Redundant Serialization

Since `_children` is fully reconstructible from parent pointer walks (as shown in `from_snapshot()` at L66–L74), it is deliberately excluded from the pickle payload. This is a clean separation of concerns: the *organisms list* is canonical state, while `_children` is a derived cache rebuilt on load. That avoids needing to keep two serialized graph representations in sync.

### LLM Context Window Constraints Drive Bounded Ancestor Views

`AncestorLearningLogView` accepts a `max_depth` parameter. Without this bound, long lineage chains could generate too many learning log entries for a mutator prompt. The parent pointer chain provides **natural depth control** — walk up `max_depth` ancestors, stop. This is a direct architectural response to LLM context limits.

***

## 8. Alternative Approaches and Why They Were Not Used

| Alternative | What It Would Require | Why It Was Rejected |
|---|---|---|
| **Re-scan on every weight compute** | O(N²) full scan of `_organisms` to count children | Too slow at population scale; burns LLM API budget on overhead |
| **Store child count as integer in Organism** | Extra mutable bookkeeping on every insertion | Child count is already derivable from `_children`, so duplicating it would add sync logic without helping graph traversal |
| **Separate `children_count` dict** | Two separate data structures to keep in sync | `_children` already provides count via `len()`, plus full graph traversal; two structures add sync risk |
| **Parent-only linked list (no `_children`)** | Sufficient for `AncestorLearningLogView` | Cannot power `NeighborhoodLearningLogView` or `FixedTreePopulation` generation frontier; novelty bonus would require O(N) scan |
| **Full graph DB (e.g., NetworkX)** | Heavy dependency, complex API | The organism graph is a DAG with simple append-only semantics; `defaultdict(list)` is sufficient and zero-dependency |

***

## 9. Architectural Diagram: Replicator Dynamics and Lineage Flow

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              EVOLVE ITERATION LOOP                      │
                    │  (evolve_problem_loop.py → evolver.py)                  │
                    └──────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────────────┐
                    │   1. SELECTION: sample_parents(k, iteration)            │
                    │      ┌──────────────────────────────────────────────┐   │
                    │      │ WeightedSamplingPopulation._compute_weights()│   │
                    │      │                                              │   │
                    │      │  for each eligible organism:                 │   │
                    │      │    σ_perf = sigmoid(score - midpoint)        │   │
                    │      │    n_i = len(_children[organism.id])  ◄──────┼───┼── O(1) lookup
                    │      │    novelty = 1 / (1 + λ * n_i)              │   │
                    │      │    weight = σ_perf * novelty                 │   │
                    │      │                                              │   │
                    │      │  P_i = weight_i / Σ(weights)                │   │
                    │      └──────────────────────────────────────────────┘   │
                    └──────────────────┬──────────────────────────────────────┘
                                       │ k parent organisms sampled
                    ┌──────────────────▼──────────────────────────────────────┐
                    │   2. LEARNING LOG CONTEXT ASSEMBLY                      │
                    │      AncestorLearningLogView:                           │
                    │        walk parent→parent→...→root (upward only)        │
                    │      NeighborhoodLearningLogView:                       │
                    │        BFS: parent + get_children(parent) ◄──────────── uses _children
                    │             + get_children(sibling) ...                 │
                    └──────────────────┬──────────────────────────────────────┘
                                       │ (organism, failure_cases, log_entries)
                    ┌──────────────────▼──────────────────────────────────────┐
                    │   3. MUTATION: LLM generates offspring                  │
                    │      ThreadPoolExecutor(mutator_concurrency=10)         │
                    │      _mutate_and_inject_attributes()                    │
                    │        → forces offspring.parent = parent_organism      │
                    └──────────────────┬──────────────────────────────────────┘
                                       │ list[Organism] (with parent pointer set)
                    ┌──────────────────▼──────────────────────────────────────┐
                    │   4. EVALUATION: score offspring                        │
                    │      evaluator.evaluate(mutated_organism)               │
                    │      → EvaluationResult(score, failure_cases)           │
                    └──────────────────┬──────────────────────────────────────┘
                                       │ Collect ALL results before writing (atomic barrier)
                    ┌──────────────────▼──────────────────────────────────────┐
                    │   5. POPULATION UPDATE: population.add(org, result)     │
                    │                                                          │
                    │      _organisms.append(...)         ← append-only log   │
                    │      _organisms_by_id[id] = ...     ← O(1) lookup index │
                    │      _learning_log.add_entry(...)   ← mutation history  │
                    │      _children[parent.id].append(child.id)  ◄───────────── LINEAGE EDGE
                    │                                                          │
                    └──────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────────────────┐
                    │   ORGANISM DAG STATE (after 2 iterations, k=2)          │
                    │                                                          │
                    │   Root (score=0.1)                                       │
                    │    ├── Child_A (score=0.4)   _children[Root]=[A,B]      │
                    │    │    └── GrandChild_X     _children[A]=[X]           │
                    │    └── Child_B (score=0.6)   _children[B]=[]  ← novel!  │
                    │                                                          │
                    │   Next iteration:                                        │
                    │     Root: n=2 → novelty=1/3 → low P                     │
                    │     Child_A: n=1 → novelty=1/2 → medium P               │
                    │     GrandChild_X: n=0 → novelty=1.0 → high P (fresh!)  │
                    │     Child_B: n=0 → novelty=1.0 → high P (fresh!)        │
                    └──────────────────────────────────────────────────────────┘
```

***

## 10. System Design Insight: Why Lineage Tracking IS the Optimizer

The `_children` defaultdict is not merely a bookkeeping artifact — it is the **core mechanism through which the population self-regulates its search strategy**. The replicator dynamic in `darwinian_evolver` is not purely fitness-proportionate (which would converge to greedy hill-climbing). It is fitness × novelty, where novelty is operationalized as the inverse of reproductive exploitation count. Every time a parent is selected for mutation, its future selection probability decreases, automatically redirecting search pressure toward unexplored branches of the solution space.

This is architecturally equivalent to **Upper Confidence Bound (UCB)** in multi-armed bandit theory: the novelty bonus $1/(1 + \lambda n_i)$ maps directly to the exploration bonus term in UCB1, where organisms with fewer samples are preferentially selected. The `_children` count is the "sample count" in this analogy. Without it, the system degenerates into either pure exploitation (greedy) or pure exploration (random), both of which are empirically worse for LLM-guided search.

The bidirectional DAG also enables the `NeighborhoodLearningLogView`'s context assembly to provide the LLM with cross-branch information, giving it visibility into "what was tried nearby and why it failed or succeeded" — a differential signal that directly improves mutation quality per iteration. The lineage graph is thus both the **population memory** and the **exploration policy** of the evolutionary optimizer.

---

