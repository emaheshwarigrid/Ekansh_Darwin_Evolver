# Learning Log Observability and Failure Case Tracking in Imbue's Darwinian Evolver (Markdown-Safe Math)

## Observation: Learning Log as a Differential Evolution Trace

Imbue's Darwinian Evolver augments the evolutionary loop with a **learning log**: a compact record of past attempted changes together with the observed outcomes of those changes. In the repository, this log is intentionally lightweight. Rather than storing full organism snapshots, raw diffs, or a telemetry table of parent/child metrics, each `LearningLogEntry` contains exactly two pieces of information:

- `attempted_change`
- `observed_outcome`

That design makes the learning log an LLM-readable trace of what was tried and how it seemed to go, which is exactly the kind of context a mutator can reuse in a later prompt.

The learning log is consulted when creating new mutations: for a given parent, the evolver selects log entries from either the organism's ancestors or from a graph neighborhood around that parent and surfaces them to the mutating LLM as contextual hints. This turns the log into a per-lineage observability layer over the evolutionary history, but in a deliberately compressed form: it exposes prior attempted changes and summarized outcomes, not a full analytical database of mutation metadata.

## Observation: Failure Cases as Targeted Feedback Signals

In addition to the learning log, the evolver tracks **failure cases** through each organism's `EvaluationResult`: concrete inputs on which the organism currently fails, optionally enriched by problem-specific fields such as outputs, error messages, or traces. These failure cases are always used to guide mutation.

They are also used for mini-evaluation **only if** post-mutation verification is enabled for that problem. In other words:

- failure cases are mandatory as mutation guidance,
- failure-case verification is optional as a compute-saving filter.

This explicit coupling between mutations and the failure cases they are supposed to fix turns the evolutionary process into a sequence of targeted experiments. The learning log captures what change was attempted and what happened overall, while the failure-case objects preserve the concrete error context that explains what the mutator was trying to repair.

## Design Evidence from Public Descriptions of the Repository

The public README describes the learning log as:

> "An entry in the learning log consists of two parts:
> 1. A summary of the change that a mutator attempted (the `attempted_change`)
> 2. An observed outcome of how this change affected the evaluation performance of the resulting organism (the `observed_outcome`)"

The same README explains that, when invoking the mutator, the system can provide a selection of learning log entries from either:

- the ancestor chain of the parent organism, or
- a bounded neighborhood around the parent in the lineage graph.

Taken together, the public design description and the repository code imply a much narrower mechanism than a full mutation-analytics store:

- a `LearningLogEntry` object that stores only `attempted_change` and `observed_outcome`,
- a write path in `Population._add_to_learning_log(...)` that creates one entry after evaluation,
- view classes that control which entries are visible during mutation.

This is important architecturally: the learning log is optimized for **prompt usefulness**, not for exhaustive telemetry.

## Design Reasoning: Why a Differential Learning Log Improves Observability

### Differential vs. Snapshot Observability

Traditional evolutionary frameworks often log only per-generation snapshots and aggregate scores, which tells you that performance changed but not which specific edit likely caused the change. Darwinian Evolver instead treats each mutation as a first-class event and logs it in compact differential form:

- what change was attempted,
- what outcome was observed after evaluating the resulting child.

Conceptually, consider a sequence of organism versions $o_0$, $o_1$, $o_2$, ... with scores $f(o_0)$, $f(o_1)$, $f(o_2)$, and so on. A learning-log entry for mutation $k$ stores:

- $m_k$: a natural-language summary of the attempted change,
- $u_k$: an observed-outcome string derived after evaluating the child relative to its parent.

The repository does **not** store a full tuple like $(m_k, f(o_k), f(o_{k+1}), \delta_f)$ in the learning log itself. Instead, it compresses that comparison into an outcome string such as:

- improved,
- worse,
- same,
- inconclusive / non-viable.

This enables:

- **Local observability**: For a specific parent, the mutator can see which nearby attempted changes looked helpful or harmful.
- **Global observability**: Across the run, developers can inspect recurring classes of attempted changes and whether they tend to correlate with good descendants.

This is closer to experiment tracking than to simple checkpointing: each mutation is represented as an experiment with a summarized result, not just as another archived organism.

### Observability Across the Population

Because the evolver maintains a population of organisms and repeatedly samples parents based on score and novelty, the learning log accumulates evidence across many lineages. At any given time, one can inspect the visible entries around a parent to see:

- which attempted changes have repeatedly failed to improve outcomes,
- which kinds of edits have previously correlated with better descendants nearby.

For long-running evolutions or large benchmarks such as ARC-AGI, this lineage-aware observability is crucial: manually diffing whole organisms across many mutations is infeasible, but reading compact summaries of "what we tried here" and "how it went" is tractable and directly useful in the next LLM prompt.

## Failure Case Tracking: Ground Truth for Mutation Effectiveness

### Failure Cases as Per-Mutation Supervision

The evolver feeds **failure cases** into the mutator: concrete inputs on which the current parent fails, together with any problem-specific fields attached by the evaluator. It can optionally expose multiple failure cases at once for mutators that support batch mutation.

For each mutation, the framework always has:

- a sampled set of trainable failure cases used to guide the mutator,
- a full evaluator run that produces the child organism's next `EvaluationResult`.

If post-mutation verification is enabled, the evaluator may also run a smaller verification check on the mutation's originating failure cases before the full evaluation. That step is optional and problem-defined rather than a mandatory part of the learning-log mechanism.

### Why Failure Tracking Is Essential for Interpreting the Learning Log

Without explicit failure-case tracking, an apparently promising mutation would be much harder to interpret: developers and mutators would know the child scored differently, but not what concrete mistakes motivated the change. By associating each mutation with a specific batch of failure cases, the system can answer much sharper questions, such as:

- Which concrete mistakes was this mutation trying to address?
- Are there recurring failure types that keep being sampled and still persist after many generations?
- When verification is enabled, did the mutation actually improve on at least one of the motivating failures?

This is only possible because the system keeps both:

- a mutation-summary channel (the learning log),
- a concrete error channel (failure cases).

Without the latter, the textual log would be far less actionable.

## Mathematical View: Fitness Deltas and Failure Fix Rates

### Fitness Delta per Mutation

Let $f(o)$ denote the scalar fitness of organism $o$. For a mutation that transforms parent $o_p$ into child $o_c$, the numeric score change is

$$
\delta_f = f(o_c) - f(o_p).
$$

The important repo-specific nuance is that the learning log does **not** store $\delta_f$ explicitly. Instead, the system uses the parent and child evaluation results to generate an `observed_outcome` string that qualitatively summarizes whether the change improved, worsened, or preserved performance.

So the architecture separates two roles:

- the **population** uses exact numeric fitness for future selection,
- the **learning log** exposes a compact natural-language interpretation of that comparison for future mutation.

#### Numeric Example: Sensitivity of Selection to Logged Outcomes

Suppose the scoring process yields:

- $f(o_p) = 0.60$,
- $f(o_c) = 0.63$.

Then

$$
\delta_f = 0.63 - 0.60 = 0.03.
$$

Assume the evolver uses a sigmoid-based sampling weight for selection of the form

$$
w(o) = \operatorname{sigmoid}(\lambda \cdot (f(o) - m)),
$$

where

$$
\operatorname{sigmoid}(z) = \frac{1}{1 + \exp(-z)}.
$$

With $m = 0.60$ and $\lambda = 10$:

$$
w(o_p) = \operatorname{sigmoid}(10 \cdot (0.60 - 0.60)) = 0.5
$$

$$
w(o_c) = \operatorname{sigmoid}(10 \cdot (0.63 - 0.60)) = \operatorname{sigmoid}(0.3) \approx 0.5744
$$

If the population temporarily contained only these two organisms, their normalized selection probabilities would be:

$$
P(\text{select parent}) = \frac{0.5}{0.5 + 0.5744} \approx 0.465
$$

$$
P(\text{select child}) = \frac{0.5744}{0.5 + 0.5744} \approx 0.535
$$

So a seemingly small $\delta_f$ already biases future selection toward the child. In the learning log, this same event would appear in compressed form as an observed outcome saying the organism improved on the parent's score. The numeric selector and the textual learning log are therefore aligned, but they are not the same data structure.

### Failure-Case Fix Rate per Mutation

For the batch of failure cases $F = \{x_1, \ldots, x_n\}$ associated with a given parent, define a binary indicator $y_i$ for each case:

- $y_i = 1$ if the child succeeds on $x_i$ where the parent failed,
- $y_i = 0$ otherwise.

The **failure-fix rate** $r$ for that mutation is then:

$$
r = \frac{1}{n} \sum y_i.
$$

If post-mutation verification is enabled, the rule "discard if no improvement on any failure case" is equivalent to retaining only mutations with $r > 0$. If verification is disabled, the mutation still proceeds to full evaluation, and the failure cases remain useful as the explanation of what the mutator was targeting.

#### Numeric Example: Effect of the Verification Threshold on Mutation Filtering

Assume two mutator types:

- $M_1$: conservative edits.
- $M_2$: aggressive edits.

For each mutation, the system uses $n = 5$ failure cases.

Empirical mutation outcomes over a run:

- For $M_1$: 10 mutations with $r_k = 0.2$ (fixing 1 of 5 failures), 5 mutations with $r_k = 0$.
- For $M_2$: 6 mutations with $r_k = 0.6$ (fixing 3 of 5 failures), 9 mutations with $r_k = 0$.

**Without** verification, all of these mutations would proceed to full evaluation.

**With** the rule "discard mutations with $r = 0$":

- For $M_1$, 10 of 15 mutations survive verification.
- For $M_2$, 6 of 15 mutations survive verification.

The effect of the verification rule is to sharpen the evaluation boundary around mutations that demonstrate at least some local corrective signal. The repository treats this as an optional efficiency/control knob, not as a required part of the learning log itself.

## Architectural Rationale: Why the System Is Structured This Way

### Constraints from LLM-Driven Mutation

LLM-based mutation is computationally expensive and inherently noisy: each mutation consumes nontrivial LLM budget, and a large fraction of naive edits will either do nothing or degrade performance. The architecture therefore needs to:

- **Maximize learning per expensive mutation** by recording concise mutation-level summaries (`attempted_change`, `observed_outcome`) that can be fed back into later prompts.
- **Optionally discard unpromising mutations early**, before spending full scoring resources, when a problem defines a useful verification step on motivating failure cases.
- **Provide rich, structured context to the LLM**, including relevant learning-log entries and current failure cases, so that it can reason about previous successes and current mistakes when proposing new edits.

The combination of learning log and explicit failure-case tracking satisfies these constraints by turning each mutation into a small, inspectable experiment whose motivating errors and qualitative outcomes can guide future mutations.

### Evolutionary and Statistical Considerations

From an evolutionary perspective, these mechanisms increase **selection pressure on informative variation** rather than random drift. The population selector still acts on numeric fitness, while the learning log and failure cases improve how mutators spend their next mutation attempts.

From a statistical perspective, the framework supports, but does not universally enforce, separating trainable and holdout failures. What it does enforce is a separation of roles:

- failure cases explain what the mutator is trying to repair,
- evaluator scores determine population selection,
- learning-log entries summarize what kind of change appeared to help or hurt.

Without explicit failure-case tracking, this observability would collapse into a much weaker signal, making it much harder to reason about mutation effectiveness.

## ASCII Architecture Diagram: Learning Log and Failure-Case Flow

```text
+----------------------+         +----------------------+         +----------------------+
|   Population Store   |         |   Learning Log DB    |         |  Failure Case Store  |
|  (organisms, scores) |         | (attempted_change,   |         | (per-task inputs,    |
|                      |         |  observed_outcome)   |         |  labels, traces)     |
+----------+-----------+         +----------+-----------+         +----------+-----------+
           |                                ^                                ^
           | select parent                  | append entry                  | sample batch
           v                                |                                |
+----------+-----------+         +----------+-----------+         +----------+-----------+
|  Parent Selector     |         |  Evaluator           |         |  Failure Sampler     |
| (score x novelty)    |         | (compute fitness,    |         | (batch failure cases |
+----------+-----------+         |  observed outcome)   |         |  for current parent) |
           |                       +----------+-----------+         +----------+-----------+
           |                                       ^                          |
           v                                       |                          |
+----------+-----------+                           |                          |
|  Mutator Orchestrator|---------------------------+--------------------------+
| (LLM call)           |  context: parent code, failure batch,
|                      |           relevant learning-log neighborhood
+----------+-----------+
           | child organism
           v
+----------+-----------+
| Post-Mutation        |
| Verification         |  optional mini-eval on failure batch
+----------+-----------+
           |
           v
+----------+-----------+
| Population Update    |
| (insert child,       |
|  update scores)      |
+----------------------+
```

This diagram shows how the learning log and failure-case tracking sit in the main evolutionary loop: every mutation is conditioned on both prior logged lessons and explicit failure examples, then globally evaluated, with concise outcomes written back for future guidance. Optional verification can add a local failure-case check before the full evaluation, but it is not required by the framework.
