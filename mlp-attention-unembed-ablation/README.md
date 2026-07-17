# Inkling MLP / attention / unembed ablation

Status: **scoped, not yet run**  
Last updated: 2026-07-16

## Question

In which Tinker-exposed broad subsystem is a small synthetic behavior easiest to install in `thinkingmachines/Inkling`?

The proposed v0 is a selective-adaptation experiment, not an activation ablation. It trains separate LoRA adapters with only MLP, only attention, only unembedding, or all three enabled. The behavior is a fictional two-hop registry that supports clean tests of memorization, paraphrase transfer, and unseen composition.

## Scope outcome

The primary comparison is deliberately narrow:

> With rank-1 LoRA, one fixed training recipe, and the same loss-bearing tokens, which subsystem learns the registry behavior fastest without damaging unrelated behavior?

This answers “easiest under this intervention.” It does **not** identify where the corresponding capability lives in the base model.

The largest design risk is adapter capacity. Current Tinker Cookbook counts for Inkling differ by roughly 747x at the same rank:

| Trainable subsystem | Parameters at rank 1 |
|---|---:|
| MLP | 154,705,920 |
| Attention | 3,424,256 |
| Unembed | 207,168 |
| All three | 158,337,344 |

For that reason, v0 reports an equal-rank result and treats a parameter-matched comparison as a gated follow-up. The closest nominal parameter match is MLP rank 1, attention rank 45, and unembed rank 747; the last rank must be confirmed as service-supported before any run is scheduled.

## Files

- [EVAL_SPEC.md](EVAL_SPEC.md) — preregistration-style experiment scope, metrics, run matrix, stopping rules, cost envelope, and interpretation limits.

Planned implementation artifacts will stay in this directory under `data/`, `configs/`, `src/`, and `results/`.

## Proposed next action

Run only the preflight gates first: verify target tokenization and renderer behavior, enumerate the tensors actually selected by each Tinker flag, confirm supported LoRA ranks, evaluate the untrained base, and validate learnability with the all-modules positive control. No subsystem winner should be reported until those gates pass.

