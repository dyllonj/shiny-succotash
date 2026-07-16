# Eval spec: coarse localization of behavior installability in Inkling

Status: **draft scope; not yet implemented or run**  
Model: `thinkingmachines/Inkling` (64K Tinker variant)  
Reviewed against public documentation: 2026-07-16

## 1. Decision this eval should support

Decide which Tinker-exposed subsystem—MLP, attention, or unembedding—is the best first target for installing a small, novel symbolic behavior with LoRA.

The result should distinguish three outcomes:

1. One subsystem learns faster and generalizes while meeting retention guardrails.
2. A subsystem memorizes direct examples but fails to generalize or causes broad output drift.
3. The evidence is task-dependent or too close to name a unique winner.

The eval is useful even in outcome 2 or 3. A clean “no unique subsystem” result is preferable to ranking configurations whose capacity or optimization conditions are not comparable.

## 2. Exact research question and estimands

### Primary estimand: protocol installability

With all data, ordering, optimizer settings, LoRA rank, renderer settings, and training-token budgets held fixed, which module-only adapter reaches the behavior criterion with the fewest cumulative loss-bearing tokens?

This is the lowest-cost and most reproducible reading of “easiest to install.” It includes the inductive bias and parameter allocation induced by Tinker's LoRA implementation.

### Secondary estimand: nominal parameter efficiency

At approximately equal trainable-parameter counts, which module-only adapter produces the strongest held-out behavior?

This comparison is gated because Inkling's per-rank counts are extremely different and the required unembed rank may not be supported. Even if feasible, equal parameter count does not equal equal effective rank, compute, update geometry, or optimizer conditioning.

### Explicit non-claim

The eval does not locate where the behavior was originally represented or “stored” in the base model. It only measures where a new behavior is easiest to add through Tinker's current selective LoRA parameterization.

## 3. Behavior to install

Use a fictional, closed-world, two-hop registry. Each generated world contains:

- 16 pronounceable nonce entities, used only as inputs.
- 8 arbitrary guide IDs.
- 8 arbitrary signal codes.
- A balanced mapping from each nonce entity to one guide ID; every guide has two entities.
- A random permutation mapping each guide ID to one signal code.
- A distinguished one-token `UNKNOWN` answer for out-of-registry entities.

Example structure, with final strings chosen only after tokenizer validation:

```text
entity daxen  -> guide G7
guide G7      -> signal cobalt
```

Example training queries:

```text
Registry query: Which guide is assigned to daxen?
G7

Registry query: Which signal belongs to guide G7?
cobalt
```

Example unseen-composition query:

```text
Registry query: Which signal belongs to the guide assigned to daxen?
cobalt
```

Facts appear only in supervised training examples. Evaluation prompts do not carry an in-context registry. All answers are answer-only, with no chain of thought.

### Why this behavior

- Random mappings make pretraining contamination testable and unlikely.
- Direct queries measure installation and memorization.
- Unseen phrasings separate exact-string memorization from transfer.
- Two-hop queries test whether the adapter supports use of the installed facts rather than merely boosting answer tokens.
- Unknown queries expose global answer-token boosting, a particularly important failure mode for unembed-only training.
- Short, fixed-choice answers make exact match and target-token margins cheap to compute.

The conclusion must remain scoped to this behavior family. A response-style or long-form reasoning behavior may rank subsystems differently.

## 4. Data construction and splits

Generate three paired synthetic worlds. A paired seed determines the world mapping, nonce strings, adapter initialization, and example order; every subsystem sees the identical world and order for that seed.

Before generation is finalized, check candidate guide IDs, signal codes, and `UNKNOWN` under the exact Inkling tokenizer and renderer. Each answer must be a single token in assistant-answer position. Reject strings with ambiguous normalization or tokenization.

### Training set per world: 64 examples

| Slice | Count | Construction |
|---|---:|---|
| Entity to guide | 32 | 16 relations, each shown in two wording templates |
| Guide to signal | 16 | 8 relations, each shown in two wording templates |
| Closed-world unknown | 16 | Negative entities disjoint from evaluation negatives |

Only assistant answer tokens receive nonzero cross-entropy weight. Keep the total number of loss-bearing tokens identical across runs.

### Evaluation set per world

| Slice | Count | Purpose |
|---|---:|---|
| Direct canonical | 24 | Acquisition of all trained relations |
| Direct paraphrase | 24 | Same relations, unseen wording |
| Composition, canonical | 16 | Unseen entity -> guide -> signal chains |
| Composition, paraphrase | 16 | Composition plus wording shift |
| Unknown / spillover | 16 | Held-out nonce entities expected to return `UNKNOWN` |
| Retention anchors | 32 | Arithmetic, common facts, simple transformations, and exact instruction following |

Template authorship must happen before model outputs are inspected. Store all world definitions and rendered token IDs so every adapter is evaluated on byte-identical inputs.

## 5. Conditions

### Competitive module-only conditions

```text
MLP only       train_mlp=True,  train_attn=False, train_unembed=False
Attention only train_mlp=False, train_attn=True,  train_unembed=False
Unembed only   train_mlp=False, train_attn=False, train_unembed=True
```

### Controls

```text
Base model     no adapter and no training
All modules    train_mlp=True, train_attn=True, train_unembed=True
```

The all-modules run is a positive control for task learnability, not a candidate for the subsystem winner.

### Primary equal-rank comparison

Use rank 1 for every trained condition, subject to the service accepting rank 1. Current Tinker Cookbook counts are:

| Condition | Rank | Trainable parameters |
|---|---:|---:|
| MLP only | 1 | 154,705,920 |
| Attention only | 1 | 3,424,256 |
| Unembed only | 1 | 207,168 |
| All modules | 1 | 158,337,344 |

This is an equal-low-rank comparison, not a parameter-matched comparison. Report both rank and parameter count beside every result.

### Gated parameter-matched follow-up

The closest integer-rank match to the minimum MLP capacity is:

| Condition | Rank | Trainable parameters | Difference from MLP rank 1 |
|---|---:|---:|---:|
| MLP only | 1 | 154,705,920 | 0.000% |
| Attention only | 45 | 154,091,520 | -0.397% |
| Unembed only | 747 | 154,754,496 | +0.031% |

Run this only if the service supports all three ranks for both training and sampling. If rank 747 is unsupported, report that there is no three-way overlapping parameter budget at the MLP minimum; do not silently substitute rank 256 or describe the result as matched.

## 6. Fixed training protocol

- Loss: supervised cross-entropy.
- Batch size: 16 examples.
- Schedule: constant learning rate, 32 optimizer steps, equivalent to eight passes over 64 examples.
- Optimizer: Tinker's AdamW-compatible optimizer with fixed betas and epsilon, zero weight decay, and the same gradient-clipping setting in every run.
- Checkpoints for eval: steps 0, 1, 2, 4, 8, 16, and 32.
- Renderer: exact Inkling renderer version pinned in the run manifest.
- Reasoning effort: fixed at the minimum supported direct-answer setting for both supervised rendering and evaluation.
- Sampling: temperature 0, one sample, short maximum output, renderer stop sequences.
- Randomness: three paired seeds; no seed or world is dropped after results are visible.

Inkling's recommended LoRA learning rate is not yet calibrated in the current Cookbook helper. Choose a common learning rate using an all-modules positive-control pilot on world 0 over the preregistered grid:

```text
3e-5, 1e-4, 3e-4
```

Select the smallest rate that shows monotonic acquisition without retention collapse or numerical instability, then freeze it for all primary conditions. A flatlining module may receive one labeled rescue run at the adjacent learning rate, but rescue results are secondary and cannot replace the fixed-recipe primary result.

## 7. Metrics

### Acquisition and generalization

- **Normalized exact match:** strip surrounding whitespace and one terminal punctuation mark; preserve all other text.
- **Raw format compliance:** response is exactly one allowed answer token plus renderer termination.
- **Correct-answer margin:** log probability of the correct one-token answer minus `logsumexp` over the fixed distractor answers.
- **Training loss:** answer-token cross-entropy by optimizer step.

Report exact match and answer margin separately for direct canonical, direct paraphrase, canonical composition, and paraphrased composition.

### Installability

- **Tokens to criterion:** cumulative loss-bearing tokens at the first checkpoint satisfying the behavior criterion.
- **Held-out AULC:** trapezoidal area under balanced held-out accuracy versus cumulative loss-bearing tokens, normalized to `[0, 1]` over the fixed budget.
- **Final held-out accuracy:** mean of direct-paraphrase accuracy and the two composition accuracies, with each slice weighted equally.

### Guardrails

- **Unknown accuracy:** exact `UNKNOWN` rate on held-out negative entities.
- **False activation:** fraction of unknown queries answered with any learned guide ID or signal code.
- **Retention accuracy delta:** adapter minus base exact match on the 32 fixed anchors.
- **Base-reference logprob delta:** change in log probability assigned to each base model's frozen deterministic reference answer.

## 8. Behavior criterion and winner rule

A behavior counts as installed at a checkpoint only if all of the following hold:

```text
direct paraphrase exact match >= 80%
mean composition exact match  >= 60%
unknown exact match           >= 90%
retention accuracy drop       <= 2 percentage points
```

Winner selection is preregistered as follows:

1. Among module-only candidates meeting the criterion in all three paired worlds, choose the one with the lowest mean tokens-to-criterion.
2. If checkpoint resolution produces a tie, use mean held-out AULC.
3. Require the pairwise direction against the runner-up to agree in all three paired worlds and the mean AULC gap to exceed 0.05.
4. Otherwise report **no unique winner** and show the slice-level tradeoff rather than forcing a rank.

If no module-only candidate meets the criterion, do not choose a winner from training-set accuracy. Report the failure modes and use the all-modules control to distinguish task/hyperparameter failure from selective-adaptation failure.

With only three worlds, all uncertainty is descriptive. Show every seed and paired difference; do not present an item-level bootstrap as if it captured between-world or training-run uncertainty.

## 9. Run matrix

### Required v0

| Stage | Runs | Notes |
|---|---:|---|
| Base evaluation | 3 | One per world; no training |
| All-modules LR pilot | 3 | Three LRs on world 0; selected run can count toward the main grid |
| Module-only main grid | 9 | 3 subsystems x 3 paired worlds |
| All-modules positive control | 3 | 3 paired worlds; world 0 selected-LR pilot is reused |

This is 14 distinct trained adapters after reusing the selected world-0 pilot, plus three base evaluations.

### Optional capacity follow-up

If all ranks pass preflight, reuse the MLP rank-1 results and add attention rank 45 plus unembed rank 747 for each world: six additional trained adapters.

## 10. Preflight and stopping rules

Do not launch the main grid until all of these pass:

1. **API flags:** create and save initialized adapters for every condition.
2. **Tensor manifest:** download initialized adapters and enumerate the exact tensor names selected by each flag. Confirm conditions are disjoint except for expected metadata and document whether Inkling's short convolutions are included under `train_attn`.
3. **Rank support:** verify ranks 1, 45, and 747 separately for creation, saving, and sampling. Rank 747 failure only blocks the optional matched comparison.
4. **Tokenizer:** verify all answer candidates are one token in rendered assistant position.
5. **No leakage:** base composition accuracy must be below 30%; otherwise regenerate the affected world and log the rejected seed.
6. **Positive control:** at least one all-modules pilot must reach 80% direct-paraphrase accuracy and 40% mean composition accuracy by step 32 without more than a 5-point retention drop.

Stop and diagnose rather than ranking subsystems if:

- The tensor manifests overlap unexpectedly or a flag selects no trainable tensor.
- No all-modules learning-rate pilot passes the positive-control gate.
- Rendered effort or answer masking differs between training and evaluation.
- A run becomes numerically unstable; adjust the common protocol and rerun all competitive conditions, not only the failed condition.
- Training-token or example-order parity cannot be reconstructed from logs.

## 11. Planned analysis artifacts

Produce:

1. Held-out accuracy versus loss-bearing tokens, one line per subsystem and individual paired seeds faintly shown.
2. Correct-answer margin versus tokens, split into direct and composition panels.
3. Final accuracy by slice, including unknown and retention guardrails.
4. A capacity table listing rank, exact trainable parameters, training tokens, selected learning rate, and final metrics.
5. A short error table separating wrong registry value, `UNKNOWN`, verbose/format failure, and unrelated answer.

Persist per-example predictions and target log probabilities. Aggregate CSVs alone are insufficient for auditing spillover and format failures.

## 12. Cost and effort envelope

Use the 64K model; the task does not benefit from 256K context. Public pricing reviewed on 2026-07-16 lists Inkling at $5.61 per million training tokens, $1.87 per million uncached prefill tokens, and $4.68 per million sampled tokens.

With 64 short examples, 32 steps, 14 trained adapters, answer-only generation, and seven evaluation checkpoints, the planning envelope is **$5–15** including pilot reruns. Replace this estimate with a tokenizer-derived count before launch. Log cached and uncached tokens separately because prefix-cache hits can materially change evaluation cost.

Expected hands-on effort after the harness exists is a few hours. The principal unknown is service support for unusual ranks, not dataset size.

## 13. Interpretation risks

- **Capacity confound:** equal rank allocates radically different trainable-parameter counts across Inkling's subsystems.
- **Optimization confound:** a common learning rate may favor one parameterization; per-module tuning answers a different question and must be labeled separately.
- **MoE asymmetry:** Tinker uses shared-outer LoRA for expert MLPs, so “MLP only” includes a large expert-specific intervention with different geometry from dense attention or unembedding.
- **Output-head shortcut:** unembed-only training may boost registry answer tokens globally. Unknown and retention slices are essential, not optional.
- **Task dependence:** a symbolic lookup/composition behavior may privilege different modules than style, refusal, tool use, or long-horizon reasoning.
- **Renderer dependence:** reasoning effort and answer masking can change the learning problem. Freeze and record both.
- **Broad labels:** `train_mlp`, `train_attn`, and `train_unembed` are API groupings, not neuron-level anatomical claims. The downloaded tensor manifest defines what was actually changed.
- **Storage fallacy:** ease of adding a behavior does not reveal where a related pretrained behavior is represented.

## 14. Out of scope for v0

- Per-layer or per-expert localization.
- Residual-stream activation patching, probing, steering, or SAEs.
- Multimodal versions of the registry.
- Full fine-tuning or base-weight modification.
- More than one behavior family.
- Claims about original feature storage.

If v0 yields a stable subsystem winner, the most informative next test is a second behavior family—such as a response-format policy—to measure whether the ranking generalizes beyond synthetic relational knowledge.

## 15. Reproducibility layout

All future work for this eval stays below this directory:

```text
mlp-attention-unembed-ablation/
  README.md
  EVAL_SPEC.md
  data/
    world_<seed>.json
    rendered_<seed>.jsonl
  configs/
  src/
  results/
    manifests/
    predictions/
    metrics/
    figures/
```

Every run manifest must include model ID, service date, SDK and Cookbook versions, renderer version, subsystem flags, rank, trainable parameter count, world seed, adapter seed, data-order seed, optimizer settings, effort value, rendered token counts, and checkpoint paths.

## 16. Primary sources

- [Tinker `ServiceClient` selective LoRA flags](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/)
- [Tinker `get_lora_param_count`](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/hyperparam_utils/get_lora_param_count/)
- [Pinned Cookbook source for parameter counts and Inkling LR status](https://github.com/thinking-machines-lab/tinker-cookbook/blob/896c411e46ff85173c9883b503511b9a2a8f0a10/tinker_cookbook/hyperparam_utils.py)
- [Tinker cross-entropy loss and target-token log probabilities](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/)
- [Tinker LoRA primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/)
- [Tinker models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- [Inkling model weights and model card](https://huggingface.co/thinkingmachines/Inkling)

