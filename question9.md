# Q9 - Verification, Mutation Validation, and Computational Efficiency



## Q9 Analyze the relationship between verification (validating mutation structure before evaluation) and computational efficiency. When would skipping verification be a critical optimization?



Verification in `darwinian_evolver` is an *optional mini‑evaluation gate* that trades extra cheap evaluations on a few failure cases for avoiding many expensive full evaluations, often yielding order‑of‑magnitude compute and cost savings on LLM‑heavy problems.[web:6] Skipping verification becomes a *critical* optimization when verification is itself expensive or uninformative (e.g., noisy evaluators, multi‑step fixes, trivially cheap full evaluations, or early exploratory phases), because in those regimes the verification gate mostly adds latency and cost while harming exploration.[web:4][web:6]  

---

## Observation: Optional Post‑Mutation Verification as a Mini‑Evaluator

The core loop is: sample parents → mutate via LLM → optionally verify each mutation on a small set of failure cases → fully evaluate successful mutations → update population.[web:4][web:6][web:13]  

Post‑mutation verification is explicitly described as an *optional* step, intended to “filter out mutations that are unlikely to yield improvements before conducting a full dataset evaluation, potentially saving both time and costs,” with reported “time and cost efficiencies improving by over tenfold” in Imbue’s experiments when this filter is predictive.[web:6]  

---

## Code Evidence: Verification Gate in `Evolver.evolve_iteration`

The verification/efficiency relationship is concentrated in the `Evolver.evolve_iteration` method, which decides whether each mutated organism should incur a full evaluation or be dropped early.  

[darwinian_evolver/darwinian_evolver/evolver.py L113‑121](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L113-L121)  
```py
def evolve_iteration(self, num_parents: int, iteration: int | None = None) -> EvolverStats:
    num_mutate_calls = 0
    num_generated_mutations = 0
    num_mutations_after_verification = 0
    num_evaluate_calls = 0
    num_verify_mutation_calls = 0
```
This sets up counters that record how many organisms are filtered by verification vs sent to full evaluation, making verification’s impact on throughput measurable.  

[darwinian_evolver/darwinian_evolver/evolver.py L155‑169](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L155-L169)  
```py
# Build futures that return (organism, passed_verification) tuples
mutated_organisms_futures = []
for mutated_organisms_future in concurrent.futures.as_completed(mutator_futures):
    mutated_organisms = mutated_organisms_future.result()
    num_generated_mutations += len(mutated_organisms)
    for mutated_organism in mutated_organisms:
        if self._should_verify_mutations:
            future = evaluator_executor.submit(self._verify_mutation, mutated_organism)
            num_verify_mutation_calls += 1
        else:
            # Without verification, all organisms are considered to pass
            future = concurrent.futures.Future()
            future.set_result((mutated_organism, True))
        mutated_organisms_futures.append(future)
```
Here, verification is a boolean gate:  
- When `_should_verify_mutations` is `True`, each mutated organism incurs an extra evaluator‑pool job to run `_verify_mutation`, incrementing `num_verify_mutation_calls`.  
- When `False`, the system short‑circuits the verification phase and treats every mutation as accepted, eliminating the extra concurrency overhead and function calls.  

[darwinian_evolver/darwinian_evolver/evolver.py L170‑181](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L170-L181)  
```py
# Then filter by passed_verification boolean
organism_evaluation_futures = []
for future in concurrent.futures.as_completed(mutated_organisms_futures):
    organism, should_evaluate = future.result()
    if should_evaluate:
        num_mutations_after_verification += 1
        evaluation_future = evaluator_executor.submit(self._evaluator.evaluate, organism)
        organism_evaluation_futures.append((organism, evaluation_future))
        num_evaluate_calls += 1
    else:
        self._population.add_failed_verification(organism)
```
The boolean result from verification directly controls whether a full `evaluate` call (often hundreds of LLM calls) is scheduled, or the organism is discarded cheaply; without verification, *all* mutations increment `num_evaluate_calls`.  

[darwinian_evolver/darwinian_evolver/evolver.py L201‑220](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolver.py#L201-L220)  
```py
def _verify_mutation(self, organism: OrganismT) -> tuple[OrganismT, bool]:
    return organism, self._evaluator.verify_mutation(organism)

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
The evolver wires verification to a problem‑specific `Evaluator.verify_mutation` method, and guarantees that each organism carries the failure cases it was mutated from; this allows verification to be implemented as a *mini‑evaluation* on exactly those failure cases.  

---

## Code Evidence: Wiring `--verify_mutations` From CLI to Evolver

The `verify_mutations` knob is a first‑class hyperparameter exposed through the CLI and propagated down to `Evolver`.  

[darwinian_evolver/darwinian_evolver/cli_common.py L13‑16](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/cli_common.py#L13-L16)  
```py
class HyperparameterConfig(BaseModel):
    batch_size: int
    verify_mutations: bool
    num_parents_per_iteration: int
    sharpness: float | None
```
`verify_mutations` is part of the hyperparameter config, not hard‑coded, so you can tune it based on cost/efficiency trade‑offs for a specific problem.  

[darwinian_evolver/darwinian_evolver/cli_common.py L55‑67](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/cli_common.py#L55-L67)  
```py
arg_container.add_argument(
    "--verify_mutations",
    action="store_true",
    default=False,
    required=False,
    help="Verify mutations before adding them to the population. Only mutations that improve on the given failure cases will be accepted.",
)
```
`--verify_mutations` is an *opt‑in* flag; by default the system skips verification, maximizing raw throughput and mutation diversity.  

[darwinian_evolver/darwinian_evolver/cli_common.py L93‑110](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/cli_common.py#L93-L110)  
```py
def build_hyperparameter_config_from_args(args: argparse.Namespace) -> HyperparameterConfig:
    return HyperparameterConfig(
        verify_mutations=args.verify_mutations,
        num_parents_per_iteration=args.num_parents_per_iteration,
    )
```
This wiring cleanly carries the CLI choice into the evolution hyperparameters.  

[EvolverProblemLoop constructor and initialization](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolve_problem_loop.py#L53-L87)  
```py
class EvolveProblemLoop:
    _should_verify_mutations: bool
    def __init__(..., batch_size: int = 1, should_verify_mutations: bool = False, ...):
        self._should_verify_mutations = should_verify_mutations
```

[EvolveProblemLoop → Evolver wiring](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/evolve_problem_loop.py#L144-L153)  
```py
self._evolver = Evolver(
    should_verify_mutations=self._should_verify_mutations,
)
```
The loop simply passes `should_verify_mutations` into `Evolver`, so the same binary choice controls every iteration.  

[darwinian_evolver/darwinian_evolver/__main__.py L160‑173](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/__main__.py#L160-L173)  
```py
evolve_loop = EvolveProblemLoop(
    should_verify_mutations=hyperparameter_config.verify_mutations,
)
```
The top‑level CLI executable hands the CLI flag all the way down to `Evolver`, making verification a runtime tuning decision that can be flipped for different workloads without changing problem code.  

---

## Code Evidence: Concrete `verify_mutation` on Multiplication Verifier

The multiplication verifier problem demonstrates a realistic `verify_mutation` that re‑runs the organism on a subset of failure cases and accepts it if at least one failure is fixed.  

[darwinian_evolver/darwinian_evolver/problems/multiplication_verifier.py L224‑262](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/multiplication_verifier.py#L224-L262)  
```py
class MultiplicationVerifierEvaluator(
    Evaluator[MultiplicationVerifierOrganism, EvaluationResult, MultiplicationVerifierEvaluationFailureCase]
):
    def evaluate(self, organism: MultiplicationVerifierOrganism) -> EvaluationResult:
        trainable_failure_cases = []
        holdout_failure_cases = []
        for data_point in self._trainable_data:
            maybe_failure_case = self._evaluate_data_point(organism, data_point)
            if maybe_failure_case is not None:
                trainable_failure_cases.append(maybe_failure_case)
        for data_point in self._holdout_data:
            maybe_failure_case = self._evaluate_data_point(organism, data_point)
            if maybe_failure_case is not None:
                holdout_failure_cases.append(maybe_failure_case)

        num_total = len(self._trainable_data) + len(self._holdout_data)
        num_correct = num_total - len(trainable_failure_cases) - len(holdout_failure_cases)
        score = num_correct / num_total
```
A full evaluation here means running the LLM on all 15 trainable + 5 holdout data points, computing a score from the proportion of correct responses. This is *expensive*, as every `_evaluate_data_point` call triggers a full LLM interaction.  

[darwinian_evolver/darwinian_evolver/problems/multiplication_verifier.py L264‑277](https://github.com/imbue-ai/darwinian_evolver/blob/main/darwinian_evolver/problems/multiplication_verifier.py#L264-L277)  
```py
def verify_mutation(self, organism: MultiplicationVerifierOrganism) -> bool:
    """Verify that the mutation of the organism has addressed at least one of the given failure cases."""
    failure_cases = organism.from_failure_cases
    assert failure_cases is not None
    for failure_case in failure_cases:
        data_point = failure_case.data_point
        maybe_failure_case = self._evaluate_data_point(organism, data_point)
        if maybe_failure_case is None:
            # If the mutation did no longer produce a failure case for this data point, we consider it a success.
            # Note that this type of verification is not entirely reliable, since some failures can originate from
            # randomness in the LLM response and might fail/pass in a given run just by chance.
            return True

    return False
```
Key points:  
- Verification reuses `_evaluate_data_point` on just the subset of failure‑case data points `organism.from_failure_cases`, not all 20 global points.  
- It returns `True` as soon as *one* previous failure disappears.  
- The comment explicitly acknowledges that verification is imperfect due to LLM randomness—an important caveat for when verification may hurt search rather than help.  

In the multiplication verifier, evaluation and verification both cost LLM calls, but full evaluation costs $20$ calls, while verification costs at most $|\text{from\_failure\_cases}|$, which is typically much smaller; that is precisely the efficiency lever.  

---

## Design Reasoning: Why Post‑Mutation Verification Improves Efficiency

The Imbue research post frames post‑mutation verification as a way to “filter out mutations that are unlikely to yield improvements before conducting a full dataset evaluation,” reporting more than 10× time and cost savings when using mini‑evaluations on failure cases as a predictor of overall improvement.[web:6]  

From an evolutionary‑systems perspective:  

- **Expensive evaluator, cheap verifier**:  
  - *Evaluator* compares an organism against a large dataset or many ARC‑AGI tasks, often involving dozens or hundreds of LLM calls, plus scoring logic.[web:4][web:11]  
  - *Verifier* re‑checks only the local neighborhood of failure cases that were given to the mutator. This is a small, targeted subset, often a low constant factor relative to the full dataset.  

- **Selection pressure vs. exploration**:  
  - Verification introduces an additional selection bottleneck: only organisms that immediately improve on some known failure cases are allowed to incur the expensive full evaluation and enter the population.[web:4][web:6]  
  - This can massively reduce the number of wasteful evaluations on clearly bad or neutral mutations, tightening the distribution of evaluated organisms around promising regions of the search space.  

- **Concurrency and resource utilization**:  
  - The evolver runs mutators and evaluators in thread or process pools; each full evaluation may block a worker on LLM calls for seconds.[web:4][web:6][web:13]  
  - Verification shifts work from “long” jobs (full evaluations) to “short” ones (mini‑evals on a few data points). Since the evaluator pool is shared, verification is essentially an early‑exit mechanism that keeps the pool from being saturated with hopeless candidates.  

The README emphasizes that verification “can significantly reduce the number of full evaluations that need to be performed, which is especially useful if full evaluations are slow and/or costly,” but warns of reduced diversity and over‑fitting risk when evaluation is noisy or when failures require multi‑step fixes.[web:4] Those warnings are exactly the regimes where *skipping* verification can become the better efficiency choice.  

---

## Mathematical Analysis: Cost Model of Verification vs. Full Evaluation

Let:  

- $M$: number of mutated organisms generated in one iteration.  
- $C_{\text{eval}}$: cost (time or dollars) of a full evaluation via `Evaluator.evaluate`.  
- $C_{\text{ver}}$: cost of a single `verify_mutation` call (mini‑evaluation).  
- $p$: probability that a mutation passes verification (i.e., fixes at least one failure case).  

### Without Verification

Every organism is fully evaluated:  
$$
C_{\text{no-ver}} = M \cdot C_{\text{eval}} \quad [1]
$$

### With Verification

Each organism first incurs `verify_mutation`, and only those that pass incur a full evaluation:  
$$
C_{\text{ver}} = M \cdot C_{\text{ver}} + p \cdot M \cdot C_{\text{eval}} \quad [2]
$$

Verification is beneficial when $C_{\text{ver}} < C_{\text{no-ver}}$, i.e.:  
$$
M \cdot C_{\text{ver}} + p \cdot M \cdot C_{\text{eval}} < M \cdot C_{\text{eval}} \quad [3]
$$
$$
C_{\text{ver}} < (1 - p) \cdot C_{\text{eval}} \quad [4]
$$

Interpretation: **verification is a win when its per‑mutation cost is smaller than the expected savings from the fraction of organisms it filters out.**  

### Concrete Numeric Example

Assume a multiplication‑verifier‑like problem:  

- Full evaluation uses 20 data points (15 trainable + 5 holdout).  
- Each data point requires one LLM call costing $0.001$ dollars.  
- So $C_{\text{eval}} = 20 \times 0.001 = 0.02$ dollars.  
- Suppose each mutation carries 4 failure cases; verification re‑evaluates those 4:  
  - $C_{\text{ver}} = 4 \times 0.001 = 0.004$ dollars.  
- Suppose only $p = 0.2$ of mutations actually fix at least one failure case (pass verification).  

**With verification:**  
$$
C_{\text{ver}} = M \cdot 0.004 + 0.2 \cdot M \cdot 0.02 = M \cdot (0.004 + 0.004) = M \cdot 0.008
$$

**Without verification:**  
$$
C_{\text{no-ver}} = M \cdot 0.02
$$

Ratio:  
$$
\frac{C_{\text{ver}}}{C_{\text{no-ver}}} = \frac{0.008}{0.02} = 0.4
$$
You save 60% of evaluation cost, i.e., a ~2.5× speedup. In practice, Imbue reports *over 10×* improvements on some problems, meaning their $C_{\text{ver}}$ was tiny and $p$ was low (many bad mutations filtered) relative to the cost of full evaluation.[web:6]  

Now flip the regime: suppose evaluation is *very cheap* but verification is almost as expensive:  

- $C_{\text{eval}} = 0.002$ (maybe just a quick local heuristic).  
- $C_{\text{ver}} = 0.0015$ (still involves several LLM calls).  
- $p = 0.8$ (most mutations that will be evaluated would also pass verification).  

Then:  
$$
C_{\text{ver}} = M \cdot 0.0015 + 0.8 \cdot M \cdot 0.002 = M \cdot 0.0031
$$
$$
C_{\text{no-ver}} = M \cdot 0.002
$$

Now $\frac{C_{\text{ver}}}{C_{\text{no-ver}}} = 1.55$: verification *increases* total cost by 55%. In such a regime, skipping verification is a critical optimization because it removes a mostly redundant pre‑filter that costs nearly as much as the full evaluation itself.  

---

## When Skipping Verification Is a Critical Optimization

Given the cost model and the repo’s design, there are several concrete scenarios where `--verify_mutations` should *not* be enabled, and skipping verification is essential for both computational efficiency and search quality.  

### 1. Evaluations Are Cheap or Bounded, Verifications Are Not

When full evaluation is already inexpensive (e.g., fast CPU‑only checks, small datasets, or trivial scoring functions), while verification still makes expensive calls (e.g., LLM‑based checks on multiple failure cases), then $C_{\text{ver}}$ can approach or exceed $C_{\text{eval}}$.  

- In that case, the inequality $C_{\text{ver}} < (1-p) C_{\text{eval}}$ fails, and verification becomes a pure overhead, adding extra latency without reducing the number of full evaluations meaningfully.  
- The multiplication verifier shows that verification reuses `_evaluate_data_point`, which always calls the LLM; if you instead have a purely symbolic or unit‑test‑style evaluator that’s cheap, simply evaluating everything is more sensible than gating via an LLM‑based verifier.  

### 2. Highly Noisy Evaluators (LLM Instability, Stochastic Tasks)

The README explicitly warns that verification “requires that evaluation results on a given data point are (mostly) consistent across runs. If the evaluation results on a given data point have high variance, then verification results cease to be indicative of a mutation’s true performance characteristics.”[web:4]  

- In such noisy regimes, a mutation may appear to “fail verification” solely due to evaluator randomness, even if it would improve the full score on average.  
- This causes the verification filter to discard potentially good stepping stone organisms, wasting the computational effort spent mutating and verifying them, and anyway forcing you to run many more iterations to achieve the same progress.  
- When the evaluator is noisy, it is often more efficient (in *progress per unit of compute*) to simply evaluate all mutations on a larger dataset and average out noise, rather than double‑sampling a small subset in verification and then sampling again in full evaluation.  

### 3. Multi‑Step Fixes Where Single Mutations Cannot Resolve Failures

The README notes that verification “requires that a single mutation step can plausibly remove a given failure case,” and warns that when problems “require a sequence of mutations before a given failure case is fully resolved, post‑mutation verification can stop those problems from making any progress.”[web:4]  

- In such landscapes, early mutations may be *directionally useful* but not yet sufficient to fix any specific failure case.  
- Verification would mark these as failures and discard them, preventing the search from traversing the necessary intermediate genotypes.  
- Skipping verification is a critical optimization here, because it allows cheap, slightly‑improving or reshaping mutations to survive to the full evaluator and, crucially, into the population, where they can accumulate across generations into real improvements.  

From an efficiency standpoint, this is subtle: the verification gate throws away compute spent on beneficial mutations that are just below the “fix at least one failure case” threshold, forcing more mutate+verify cycles. Skipping verification lets the *selection function* (fitness score) handle these nuances directly, making better use of each LLM call.  

### 4. Early‑Phase Exploration and Diversity‑Sensitive Regimes

Verification is a strong local filter that heavily biases toward “fix the known failures right now.” While that’s great for exploitation, it can be counterproductive during early exploration, where you want diversity, novelty, and broad coverage. The README warns that verification “can reduce the diversity of organisms in the population and make it harder to escape a local optimum.”[web:4]  

Situations where skipping verification is especially important:  

- **Cold start**: In the first few iterations, you may want to accept a wide variety of mutations and let the weighted‑sampling population model discover promising directions; verification might prematurely prune whole branches.  
- **Highly multimodal fitness landscapes**: You may need to traverse low‑fitness but structurally novel regions; verification would suppress these because they don’t immediately fix existing failures.  

In those cases, skipping verification is an efficiency optimization in terms of *search progress per evaluation*—you spend evaluations on a more diverse set of organisms, which can yield higher long‑term gains than aggressively filtering for local fixes.  

### 5. Concurrency Saturation and System Throughput

Because verification runs in the same evaluator executor pool as full evaluations, enabling it effectively doubles the number of evaluator jobs (first verification, then evaluation for the survivors).  

- If the evaluator pool is already the bottleneck (e.g., many parallel tasks hitting a rate‑limited LLM endpoint), then adding verification can worsen queuing delays and wall‑clock latency, even if it reduces the number of full evaluations.  
- When your main constraint is *latency* and not *total LLM tokens*, and evaluation datasets are modest, skipping verification can improve throughput: each mutated organism goes straight to evaluation without the additional round‑trip.  

---

## Relationship Summary: Verification vs. Computational Efficiency

Putting it together in the cost model and the code:  

- Verification adds a per‑mutation overhead $C_{\text{ver}}$ but reduces the expected number of full evaluations from $M$ to $pM$.  
- When $C_{\text{ver}}$ is tiny and $p$ is small, verification yields huge savings; this is the regime Imbue emphasizes (LLM‑heavy evaluators on ARC‑AGI tasks).[web:6][web:11]  
- When either:  
  - $C_{\text{ver}} \approx C_{\text{eval}}$ (e.g., both are LLM‑heavy),  
  - or $p$ is large (most mutations pass verification),  
  - or the evaluator is cheap,  
  verification becomes a net negative on compute and latency.  

In those non‑ideal regimes, the code already lets you turn verification off via `--verify_mutations`, and in fact defaults to *off* so that the framework is safe and efficient for cheap or noisy problems out of the box.  

---

## ASCII Flow Diagram: Mutation, Optional Verification, and Evaluation

```text
+---------------------------+
|  Population (organisms)   |
+---------------------------+
             |
             | sample_parents()
             v
+---------------------------+
| Parents + EvalResults     |
+---------------------------+
             |
             | for each parent:
             v
+---------------------------+
| Sample failure_cases      |
| (trainable subset)        |
+---------------------------+
             |
             | submit to mutator_executor
             v
+---------------------------+
|   Mutators (LLM)          |
|   mutate(parent,          |
|           failure_cases,  |
|           learning_log)   |
+---------------------------+
             |
             | mutated_organisms
             v
+------------------------------------------+
| For each mutated_organism:               |
|                                          |
|  if should_verify_mutations:             |
|      submit _verify_mutation()          |
|      -> verifier in evaluator_executor   |
|  else:                                   |
|      mark should_evaluate = True         |
+------------------------------------------+
             |
             | (organism, should_evaluate)
             v
+------------------------------------------+
| If should_evaluate:                      |
|   submit evaluator.evaluate(organism)    |
|   -> full dataset, compute score         |
| Else:                                    |
|   population.add_failed_verification()   |
+------------------------------------------+
             |
             | wait for all evaluations
             v
+---------------------------+
| population.add(organism,  |
|                eval_result)|
+---------------------------+
             |
             v
+---------------------------+
| Next iteration / snapshot |
+---------------------------+
```

---

## System Design Insight: Why It’s Architected This Way

Imbue’s Evolver is intended as a *universal optimizer* for any code/text problem where evaluation can be expensive and noisy.[web:4][web:6][web:8] That leads to several architectural constraints that explain this design:  

1. **Problem‑Agnostic Efficiency Lever**  
   - Different problems have wildly different evaluator costs and noise profiles.  
   - Making verification opt‑in and fully problem‑defined (`Evaluator.verify_mutation`) allows each user to decide:  
     - *What* mini‑evaluation should be (which subset of data, which heuristics), and  
     - *Whether* it’s worth running at all.  

2. **Decoupling Search Policy from Evaluation Semantics**  
   - `Evolver` treats verification as a binary gating function returning `(organism, bool)` and doesn’t care *how* that bool is computed.  
   - This keeps the core evolutionary logic independent of the domain while still enabling sophisticated cost‑saving strategies such as LLM mini‑evals on failure cases.[web:6]  

3. **Concurrency‑Friendly Early Filtering**  
   - The use of thread/process pools for both mutators and evaluators allows the system to pipeline LLM calls, but the real bottleneck is typically evaluator capacity.  
   - By running verification in the evaluator pool and using it to cut off many candidates early, they keep the pool focused on promising organisms when evaluation is expensive, while still allowing you to disable verification entirely when evaluation is cheap or you need maximum diversity.  

4. **Explicit Acknowledgment of Noisy/Non‑Local Fitness**  
   - The multiplication verifier’s comment about LLM randomness, and the README’s caution on high‑variance data and multi‑step fixes, show that the authors *expect* regimes where verification is not reliable.[web:4]  
   - Rather than forcing verification as a hard requirement, the framework exposes a simple switch (`--verify_mutations`) and problem‑side implementation hook so practitioners can *opt out* whenever verification harms progress or efficiency.  

In short: **verification is a powerful but situational efficiency tool**, architected as an optional, problem‑defined mini‑evaluator. It offers huge compute savings when full evaluations are heavy and relatively stable, but the framework is intentionally designed so that *skipping* verification is easy and, in many real problems (cheap tests, noisy LLM evaluators, or multi‑step failures), is the crucial optimization for both computational efficiency and evolutionary progress.[web:4][web:6][web:13]
