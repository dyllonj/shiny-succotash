# Quick interpretability experiments for Thinking Machines Lab's Inkling

_Release reviewed: July 15, 2026_

## Recommendation

The strongest quick experiment is an **effort × causal-chain-of-thought study**, followed by a tiny supervised fine-tuning intervention. Inkling is unusually well suited because reasoning effort is continuously controllable and Tinker provides answer-token log probabilities plus an existing True-Thinking Score implementation.

## Release summary

- **Model:** Inkling, released July 15, 2026.
- **Size:** 975B total parameters, 41B active, across 66 decoder layers.
- **MoE layout:** Each token uses 6 of 256 routed experts plus 2 always-on shared experts.
- **Attention:** Sliding-window and global layers at a 5:1 ratio, with relative positional embeddings rather than RoPE and short convolutions in the attention and residual paths.
- **Context:** Up to 1M tokens natively; Tinker currently offers 64K and 256K variants.
- **Modalities:** Text, image, and audio inputs with text output. Pretraining also included video data.
- **Training:** 45T multimodal tokens, synthetic-data SFT initialization, and more than 30M asynchronous RL rollouts.
- **Reasoning control:** A continuous effort scalar in `[0, 1)`.
- **License and weights:** Apache 2.0; BF16 and NVFP4 weights are available on Hugging Face.
- **Positioning:** Thinking Machines explicitly says Inkling is not the strongest overall model; it is intended to be a broad, customizable foundation.

Primary sources:

- [Official Inkling announcement](https://thinkingmachines.ai/news/introducing-inkling/)
- [Official model card](https://thinkingmachines.ai/model-card/inkling/)
- [Hugging Face weights](https://huggingface.co/thinkingmachines/Inkling)

## Practical constraints

Inkling is open-weight but not lightweight. Self-hosting requires approximately:

- 2 TB aggregate VRAM for BF16
- 600 GB aggregate VRAM for NVFP4

On Tinker, use:

```text
thinkingmachines/Inkling
thinkingmachines/Inkling:peft:262144
```

The first identifier provides 64K context; the second provides 256K context. The release pricing table lists the 64K variant at $1.87 per million prefill tokens, $4.68 per million sampled tokens, and $5.61 per million training tokens.

See [Tinker models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/).

The documented Tinker API provides sampling, target-token log probabilities, top-k distributions, LoRA training, downloadable adapters, and aggregate MoE load statistics. It does not expose residual-stream activations or documented per-token router traces, and it does not expose full-vocabulary logits. Therefore, the best lightweight methods are behavioral interventions, causal reasoning-step perturbations, and tiny LoRA ablations. SAE, activation-patching, and conventional probing work would require self-hosting or a new instrumentation endpoint.

## Ranked experiment list

| Experiment | Relative effort | Main question |
|---|---:|---|
| Effort × True-Thinking Score | Very low | Does extra reasoning causally help, or merely add prose? |
| Concise-vs-verbose SFT | Low | Does SFT improve computational efficiency or only change style? |
| LoRA-gradient router fingerprint | Very low, experimental | Can sparse gradient support reveal routed MoE experts? |
| MLP/attention/unembed ablation | Low | In which broad subsystem is a behavior easiest to install? |
| Cross-modal transfer | Low | Do text, image, and audio converge on a shared semantic substrate? |

## 1. Does more effort mean more real thinking?

Use 10–20 exact-answer math or logic problems and sweep:

```text
effort = 0.0, 0.3, 0.6, 0.9, 0.99
temperature = 0
```

Keep the prompt and maximum generation budget fixed. For every trace, record:

- Answer accuracy
- Correct-answer log-odds against plausible distractors
- Reasoning-token count
- Mean True-Thinking Score per step
- Fraction of steps with TTS at least 0.3
- Causal density: summed TTS per 1,000 reasoning tokens
- Fraction of self-correction steps such as “Wait” or “let me check” that are decorative

TTS perturbs individual reasoning steps and measures the change in correct-answer probability. Tinker already has a [True-Thinking Score recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/true-thinking-score/) built on `compute_logprobs`. Start with its five-problem smoke test, point it at `thinkingmachines/Inkling`, and thread Inkling's effort value through the renderer. Inkling's presets and supervised-rendering behavior are documented in [Thinking effort](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/).

This directly tests the release's interesting claim that RL made Inkling's reasoning more compressed without explicitly rewarding grammatical compression. Does high effort add high-TTS steps, or merely more narration?

### Controls

- Compare destructive step perturbations with meaning-preserving paraphrases to control for fluency sensitivity.
- Use answer log-odds against fixed distractors rather than raw probability alone.
- Give every effort condition a sufficiently large `max_tokens` value so truncation does not masquerade as low reasoning ability.
- Analyze correct and incorrect traces separately.
- If sampling above temperature zero, use multiple seeds rather than interpreting one trace.

### Interpretation ceiling

A high TTS establishes that emitted reasoning affects subsequent decoding. It does not prove that the prose faithfully reports a hidden computation that existed before the text was generated.

## 2. Can tiny SFT compress real reasoning?

Take 32–64 successful traces and create two matched adapters:

1. **Concise:** Preserve every mathematical operation but remove discourse filler.
2. **Verbose:** Preserve the same operations and final answers while adding redundant explanations and self-checks.

Use rank-4 or rank-8 LoRA, identical random seeds, identical optimizer settings, and equal total loss-bearing tokens. When rendering SFT examples, use the same effort setting used to generate them.

Rerun experiment 1 on held-out problems at every effort level. The most informative plot is:

```text
accuracy versus generated tokens, colored by causal density
```

Possible outcomes:

- Length changes while TTS and accuracy remain fixed: primarily a style intervention.
- Concise SFT raises TTS per token without hurting accuracy: evidence of genuine reasoning compression.
- Verbose SFT lowers causal density: SFT can install decorative reasoning.
- Effects appear only at the trained effort: the effort controller and reasoning policy are entangled.

This is a compact one-afternoon experiment with a plausible short paper inside it.

## 3. Recover MoE routing through LoRA gradients

This is speculative but potentially the most novel experiment.

Tinker uses a shared-outer LoRA scheme for MoE layers: one factor is shared across experts and the other remains expert-specific. Expert-specific adapter updates may therefore act as a routing side channel. See [`get_lora_param_count`](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/hyperparam_utils/get_lora_param_count/).

### Pilot design

1. Create a rank-1, MLP-only LoRA with a fixed seed.
2. Save the initialized adapter.
3. Train on one raw input token predicting one target token, with exactly one loss-bearing position.
4. Take one or two Adam steps with zero weight decay and save again.
5. Diff the expert-specific tensors by layer.

If the side channel works, nonzero update support should identify approximately six routed experts per layer.

### Validation

- Same input, different prediction target: expert support should remain similar because routing is determined while processing the input.
- Different input, same target: support should change.
- Repeat exact inputs across fresh clients with the same seed to test reproducibility.
- Verify the number of changed routed-expert tensors against Inkling's top-6 architecture.
- Pilot both one and two optimizer steps because LoRA's zero/random factor initialization may delay changes in one factor until the second step.

After validating text tokens, compare numbers, code tokens, natural-language words, minimal image patches, short audio, and different effort prefixes.

If Adam or the adapter format obscures support, use the aggregate statistics returned by `forward_backward`: expert coverage, oversubscription, and maximum load violation. These are documented in [`ForwardBackwardOutput`](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/forwardbackwardoutput/). Match sequence length and batch size exactly because expert coverage grows mechanically with token count.

Tinker adapters can be downloaded and inspected without downloading the base model; see [Build LoRA Adapter](https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/).

## 4. Coarse causal localization of an installed behavior

Teach a small synthetic behavior using nonce entities, so the base model cannot answer from pretraining. Train four adapters:

```text
MLP only       train_mlp=True,  train_attn=False, train_unembed=False
Attention only train_mlp=False, train_attn=True,  train_unembed=False
Unembed only   train_mlp=False, train_attn=False, train_unembed=True
All modules    train_mlp=True,  train_attn=True,  train_unembed=True
```

These switches are supported by the [Tinker ServiceClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/).

Evaluate:

- Exact memorization of training examples
- Paraphrased queries
- Unseen compositions of learned rules
- Unrelated-task retention
- Change in held-out token log probabilities

Match trainable parameter counts rather than simply matching rank, and use several seeds. When inspecting adapters, compare reconstructed updates `delta_W = B @ A`, not the raw A and B factors, because the factorization is not unique.

### Interpretation ceiling

This shows where a behavior is easiest to install under a particular LoRA parameterization. It does not necessarily identify where the original behavior was stored in the base model.

## 5. Cross-modal semantic transfer

Create approximately 30 fictional facts such as:

```text
A daxen's warning color is cobalt.
```

Represent each fact in three equivalent forms:

- Text
- Text rendered into an image
- 16 kHz spoken audio

Train on one modality and test the other two. Include counterfactual versions in which one color, digit, or relation changes. Combine this with the subsystem ablation or router fingerprint.

Useful measurements:

- Exact-answer accuracy and log-odds
- Zero-shot transfer between modality pairs
- Number of SFT examples required before transfer appears
- Expert-utilization overlap across modalities
- Effect of MLP-only versus attention-only adaptation

Strong transfer supports a shared downstream semantic representation. Weak transfer or disjoint expert fingerprints suggests modality-specific processing persists deeper into the decoder. This is well targeted to Inkling because its modalities are lightly embedded and then processed jointly by the transformer.

## Suggested first session

1. Run a five-problem TTS smoke test.
2. Run a 20-problem effort sweep.
3. Train two 32-example concise/verbose adapters.
4. Repeat the effort sweep and TTS analysis.
5. Produce three plots:
   - Accuracy versus generated tokens across effort
   - Causal density versus effort
   - Change in both curves after concise and verbose SFT

At the listed 64K rates, keeping traces short should put this in the single-digit to low-tens-of-dollars range. Actual cost will depend mainly on chain-of-thought length, repeated TTS forward passes, and prefix-cache hits.

## Important caveats

- TTS is causal at the level of emitted tokens and final predictions, not hidden-state mechanisms.
- LoRA update locations measure ease of intervention, not necessarily original feature location.
- Cross-modal transfer does not by itself prove shared neurons or experts.
- Tinker does not expose the residual stream needed for activation patching, linear probes, steering-vector extraction, or sparse autoencoders.
- Inkling-Small is only previewed for now; its 276B-total, 12B-active weights are expected later and may be a more practical target for activation-level work.
