# LoRA-gradient router fingerprinting on Inkling

_Project: Inkling interpretability_

_A remote, checkpoint-diff experiment for recovering routed-expert membership without loading the 975B-parameter base model._

**Status:** research protocol, not a confirmed result

**Last checked:** 2026-07-16

**Model:** `thinkingmachines/Inkling`

## Bottom line

This is worth a small pilot. The clean version is not really “reading gradients” and it does not recover the router’s weights. It is an **optimization side channel**:

1. create a rank-1, MLP-only LoRA on Inkling;
2. save its untouched adapter;
3. backpropagate one one-token example and take one AdamW step;
4. save the adapter again;
5. diff the **expert-specific** LoRA slices;
6. interpret the changed expert indices as the token’s routed-expert set at each sparse layer.

Inkling has 66 transformer layers. The released config makes the first two MLPs dense and the remaining 64 sparse. Each sparse layer chooses 6 of 256 routed experts and also evaluates 2 shared experts. If the side channel works cleanly, one token should therefore yield a fingerprint containing roughly:

```text
64 sparse layers × 6 routed expert IDs = 384 (layer, expert) memberships
```

The experiment is compute-light—one token, one backward pass, one optimizer step—but checkpoint-I/O-moderate. Tinker’s current parameter-count table says rank-1 MLP-only Inkling LoRA has **154,705,920 trainable values**. That is about **295 MiB at BF16** or **590 MiB at FP32 per checkpoint**, before container overhead. The minimum useful pilot downloads an initial and a trained checkpoint.

The key unknown is whether the first optimizer step changes exactly six expert-specific slices per sparse layer. The raw adapter format makes that plausible, but initialization, quantization, target-dependent zero gradients, or backend details could blur it. Treat the first run as a feasibility gate.

## What this would and would not recover

If successful, it recovers:

- a set of six routed expert IDs for each sparse layer;
- a reproducible “routing fingerprint” for a one-token input;
- similarity relationships between tokens, measured layer by layer;
- indirect evidence about where routing becomes semantic or task-specific.

It does **not** recover:

- router weight matrices;
- the top-6 ordering—the implementation requests unsorted top-k indices;
- router logits, sigmoid scores, mixture weights, or margins;
- causal proof that an expert implements a human-readable concept;
- a clean per-token trace for a multi-token sequence;
- the two shared experts as a routing choice, because both are always active.

Most importantly, **Adam update magnitude is not a router score**. The first Adam step approximately normalizes each nonzero gradient to its sign, so update support may preserve “used versus unused” while destroying most magnitude information.

## Why the signal might exist

### Inkling’s router

For hidden state \(h_\ell\) at sparse layer \(\ell\), the released implementation computes router logits and chooses six routed experts:

\[
r_\ell = G_\ell h_\ell
\]

\[
S_\ell(h) = \operatorname{TopK}_6\left(\sigma(r_{\ell,\text{routed}}) + b_\ell\right)
\]

The chosen six and both shared experts are then jointly normalized and scaled before their outputs are mixed. Schematically:

\[
y_\ell = \sum_{e \in S_\ell(h)} \alpha_{\ell,e} f_{\ell,e}(h)
       + \sum_{s=1}^{2} \gamma_{\ell,s} f^{\text{shared}}_{\ell,s}(h)
\]

For an ordinary expert parameter \(\theta_{\ell,e}\), an unselected routed expert is absent from that token’s computation:

\[
e \notin S_\ell(h)
\quad\Longrightarrow\quad
\frac{\partial L}{\partial \theta_{\ell,e}} = 0
\]

For a batch containing exactly one token and one loss-bearing position, the nonzero support of expert-parameter gradients should therefore identify the selected set—subject to the caveats below.

### Tinker’s shared-outer MoE LoRA

Tinker documents a “shared-outer” scheme for MoE LoRA: the factor connected to the model hidden dimension is shared across experts, while the other factor remains expert-specific.

The raw checkpoint converter documents expert factors with these shapes:

```text
lora_A: (num_experts_a, rank, in_dim)
lora_B: (num_experts_b, out_dim, rank)
```

One factor can have expert-axis size `1`; the other can have size `256`. Conversion broadcasts the size-1 factor across experts.

That means:

- the **shared factor** aggregates gradients from all selected experts and cannot reveal expert identity;
- the **256-slice factor** is the potential fingerprint carrier;
- raw checkpoint inspection must happen **before** PEFT conversion broadcasts the shared factor.

For gate/up projections, the input dimension is the 6,144-wide hidden state, so the likely layout is shared `A` and expert-specific `B`. For the down projection, the output dimension is the hidden state, so the likely layout is expert-specific `A` and shared `B`. This orientation is an inference from the documented shape convention and sharing rule; verify it from the actual raw checkpoint rather than hard-coding it.

### Initialization decides which step is informative

For one expert matrix, standard LoRA is:

\[
W'_e = W_e + c B_e A_e
\]

If `A` is randomly initialized and `B` starts at zero, then on the first backward pass:

\[
\frac{\partial L}{\partial B_e}
= c\frac{\partial L}{\partial W'_e}A_e^\top
\]

can be nonzero, while:

\[
\frac{\partial L}{\partial A_e}
= cB_e^\top\frac{\partial L}{\partial W'_e}
= 0
\]

Therefore:

- if zero-initialized `B` is expert-specific, its **first-step delta** may reveal the six selected experts cleanly;
- if zero-initialized `B` is shared, the expert-specific `A` may not move until the **second** step;
- if the implementation uses a different initialization, the observed zero-step tensors decide which factor to inspect.

Do not assume Tinker uses the textbook initialization: inspect it.

### Why one AdamW step can preserve support

Tinker’s public optimizer is AdamW. At the first step, with no weight decay and no prior momentum, each nonzero coordinate is approximately updated as:

\[
\Delta\theta_i \approx -\eta\frac{g_i}{|g_i|+\epsilon}
\]

Thus:

- exactly zero gradients remain unchanged;
- nonzero gradients tend toward an update near \(\pm\eta\);
- support can survive;
- relative magnitudes mostly do not.

Set `weight_decay=0.0` explicitly. Otherwise decoupled weight decay changes randomly initialized slices even when their task gradient is zero, creating false positives.

## The hard feasibility gates

Do these in order. Stop if a gate fails instead of scaling up a broken signal.

### Gate 1: a zero-step sampler checkpoint is downloadable

Immediately after client creation, call `save_weights_for_sampler`, download the result, and confirm it contains:

```text
adapter_model.safetensors
adapter_config.json
```

Use a sampler checkpoint, not `save_state`: the latter also exists for training resumption and may include optimizer state that this experiment does not need.

If the service refuses to save before an optimizer step, create two clients with the same seed: leave one untouched as the reference and train the other. Gate 3 then becomes mandatory.

### Gate 2: raw routed-expert factors are structurally visible

In the raw safetensors file, look for keys containing routed `.experts.` weights, but not `.shared_experts.`. For those keys, confirm:

- tensors are 3-D;
- the first axis is `1` or `256`;
- at least one factor in each relevant pair has first-axis size `256`;
- layers 0 and 1 are dense, while layers 2 through 65 expose routed-expert tensors.

If all exported factors are flattened, merged, or already broadcast without a way to identify the original expert-specific factor, stop: this checkpoint representation cannot carry a clean fingerprint.

### Gate 3: the baseline is deterministic

Create two untouched rank-1 clients using the same seed, save both, and diff their raw adapter tensors.

Ideal result:

```text
same keys + same shapes + bit-identical values
```

If values differ, prefer a before/after pair from the **same** client. If that is impossible, measure the null-difference distribution and require trained deltas to clear it by a large margin.

### Gate 4: one step produces sparse expert-axis support

Run one token through one backward pass and one optimizer step. For every raw tensor whose shape begins with `256`, compute an L2 norm over each expert slice after subtracting the initial tensor.

The clean signature is:

```text
most sparse layers: exactly 6 changed expert slices
dense layers: no 256-expert tensor
shared factor: may change, but is excluded
shared experts: excluded
```

If all 256 slices change, the usual causes are:

- you inspected a PEFT-converted, broadcast factor;
- weight decay was nonzero;
- you thresholded absolute weights instead of the before/after delta;
- a supposedly expert-specific factor is actually shared or coupled;
- serialization or optimizer behavior introduces background movement.

If no slices change, inspect which factor starts at zero. The first-step gradient may land only in the shared factor; try the controlled two-step fallback described later.

## Minimal end-to-end pilot

### 1. Make an isolated environment

The current Inkling extra in Tinker Cookbook requires Tinker `>=0.23.0`.

```bash
cd inkling-interpretability
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'tinker-cookbook[inkling]' safetensors
```

Set your API key without printing it:

```bash
export TINKER_API_KEY='...'
```

Do not put the key in the script, Markdown, shell history, or committed metadata.

### 2. Run one raw-token probe

This is a deliberately plain `ModelInput`, not a chat-rendered conversation. A chat renderer adds system, role, and delimiter tokens, turning the update into a union over many token routes.

Save the following as `run_router_probe.py` if you want to execute it:

```python
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import tinker
import torch
from tinker import TensorData
from tinker_cookbook import weights

MODEL = "thinkingmachines/Inkling"
SEED = 20260716

# This is the SDK's AdamParams default, not a calibrated Inkling recommendation.
# Tinker Cookbook currently says Inkling's recommended LR is uncalibrated.
LR = float(os.environ.get("ROUTER_FP_LR", "1e-4"))


def exactly_one_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {text!r}; got {len(ids)}: {ids}")
    return ids[0]


tag = f"router-fp-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
out = Path("router_fp_runs") / tag
out.mkdir(parents=True, exist_ok=False)

service = tinker.ServiceClient(
    user_metadata={"experiment": "inkling-router-fingerprint", "tag": tag}
)
train = service.create_lora_training_client(
    base_model=MODEL,
    rank=1,
    seed=SEED,
    train_mlp=True,
    train_attn=False,
    train_unembed=False,
)
tokenizer = train.get_tokenizer()

# Replace these strings freely, but keep both assertions at one token.
x_text = "7"
y_text = " A"
x_id = exactly_one_token(tokenizer, x_text)
y_id = exactly_one_token(tokenizer, y_text)

datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints([x_id]),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(
            torch.tensor([y_id], dtype=torch.long)
        ),
        "weights": TensorData.from_torch(
            torch.tensor([1.0], dtype=torch.float32)
        ),
    },
)

# Gate 1: save untouched adapter weights.
init_result = train.save_weights_for_sampler(
    name=f"{tag}-init", ttl_seconds=3600
).result()

# Exactly one loss-bearing position, then exactly one optimizer step.
fb = train.forward_backward([datum], loss_fn="cross_entropy").result()
train.optim_step(
    tinker.AdamParams(
        learning_rate=LR,
        beta1=0.9,
        beta2=0.95,
        eps=1e-12,
        weight_decay=0.0,
        grad_clip_norm=0.0,
    )
).result()

step1_result = train.save_weights_for_sampler(
    name=f"{tag}-step1", ttl_seconds=3600
).result()

weights.download(tinker_path=init_result.path, output_dir=str(out / "init"))
weights.download(tinker_path=step1_result.path, output_dir=str(out / "step1"))

metadata = {
    "model": MODEL,
    "seed": SEED,
    "rank": 1,
    "learning_rate": LR,
    "input_text": x_text,
    "input_id": x_id,
    "target_text": y_text,
    "target_id": y_id,
    "init_tinker_path": init_result.path,
    "step1_tinker_path": step1_result.path,
    "forward_backward_metrics": fb.metrics,
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
print(out)
```

Run it:

```bash
ROUTER_FP_LR=1e-4 python run_router_probe.py
```

Why this construction is unusually clean:

- input length is exactly one;
- target length is exactly one;
- exactly one position has loss weight `1.0`;
- batch size is one;
- only MLP LoRA is enabled;
- rank is the minimum useful rank;
- weight decay is explicitly disabled;
- the reference and trained weights come from the same client.

The target token affects the backward signal, but not the forward router decision. A later control checks that changing only the target preserves the recovered support.

### 3. Inspect the raw checkpoint before converting anything

Do **not** call `build_lora_adapter` first. The converter intentionally expands a size-1 expert factor into per-expert tensors, which erases the easiest clue about which factor was shared. Work directly with the downloaded raw `adapter_model.safetensors`.

Use this inspector:

```python
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

N_EXPERTS = 256


def find_adapter(root: str) -> Path:
    hits = list(Path(root).rglob("adapter_model.safetensors"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one adapter under {root!r}; found {hits}")
    return hits[0]


def layer_number(key: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)", key)
    return int(match.group(1)) if match else None


init_path = find_adapter(sys.argv[1])
step_path = find_adapter(sys.argv[2])

factor_sets: dict[int, list[set[int]]] = defaultdict(list)
rows = []

with safe_open(str(init_path), framework="pt", device="cpu") as before, \
     safe_open(str(step_path), framework="pt", device="cpu") as after:
    before_keys = set(before.keys())
    after_keys = set(after.keys())
    if before_keys != after_keys:
        raise RuntimeError("Checkpoint key sets differ")

    for key in sorted(before_keys):
        w0 = before.get_tensor(key)
        w1 = after.get_tensor(key)
        if w0.shape != w1.shape:
            raise RuntimeError(f"Shape changed for {key}: {w0.shape} -> {w1.shape}")

        is_routed_expert = ".experts." in key and ".shared_experts." not in key
        if not is_routed_expert or w0.ndim != 3 or w0.shape[0] != N_EXPERTS:
            continue

        delta = w1.float() - w0.float()
        norms = torch.linalg.vector_norm(delta.flatten(1), dim=1)

        # With same-client before/after checkpoints and weight_decay=0, unchanged
        # slices should normally be bit-identical. Replace 0.0 with a threshold
        # derived from the zero-step null control if they are not.
        threshold = 0.0
        changed = torch.nonzero(norms > threshold, as_tuple=False).flatten().tolist()
        layer = layer_number(key)

        rows.append(
            {
                "key": key,
                "layer": layer,
                "shape": list(w0.shape),
                "changed_count": len(changed),
                "changed_experts": changed,
                "max_delta_norm": float(norms.max()),
            }
        )
        if layer is not None and changed:
            factor_sets[layer].append(set(changed))

for row in rows:
    print(
        f"layer={str(row['layer']):>2} changed={row['changed_count']:>3} "
        f"shape={tuple(row['shape'])}  {row['key']}"
    )

summary = {}
for layer, sets in sorted(factor_sets.items()):
    union = set().union(*sets)
    intersection = set.intersection(*sets)
    summary[layer] = {
        "factor_count": len(sets),
        "union": sorted(union),
        "intersection": sorted(intersection),
    }
    print(
        f"L{layer:02d}: union({len(union)})={sorted(union)}  "
        f"intersection({len(intersection)})={sorted(intersection)}"
    )

Path("fingerprint_debug.json").write_text(
    json.dumps({"factors": rows, "layers": summary}, indent=2, sort_keys=True)
)
```

Run it using the directory printed by the probe script:

```bash
python inspect_router_probe.py \
  router_fp_runs/RUN_ID/init \
  router_fp_runs/RUN_ID/step1
```

First inspect each factor separately. Do not immediately trust the union:

- an informative first-step factor should usually show six changed slices;
- a factor blocked by zero initialization may show zero;
- a broadcast/shared or contaminated factor may show all 256;
- union and intersection are meaningful only across factors already judged informative.

Once the informative key pattern is known, encode that exact pattern in the analysis rather than accepting every 256-axis tensor.

## The minimum convincing control matrix

After the first probe passes, create a fresh same-seed client for each row. Each client gets one example and one optimizer step.

| ID | Input token | Target token | Purpose |
|---|---|---|---|
| `Z0a` | none | none | untouched baseline |
| `Z0b` | none | none | same-seed null/determinism control |
| `R1a` | `x1` | `y1` | primary probe |
| `R1b` | `x1` | `y1` | end-to-end repeatability |
| `T1` | `x1` | `y2` | target-invariance control |
| `X1` | `x2` | `y1` | input-discriminability control |

Use the same:

- model;
- rank;
- LoRA seed;
- optimizer settings;
- checkpoint format;
- token count;
- batch size.

Expected relationships for layerwise expert sets \(S_\ell\):

```text
S(R1a) ≈ S(R1b)              repeatability
S(R1a) ≈ S(T1)               target invariance
S(R1a) differs from S(X1)    input sensitivity
delta(Z0a, Z0b) = 0          deterministic null
```

Use Jaccard similarity per layer:

\[
J_\ell(A,B)=\frac{|S_\ell(A)\cap S_\ell(B)|}{|S_\ell(A)\cup S_\ell(B)|}
\]

Report the distribution over 64 sparse layers, not only one global average. A random independent pair of 6-of-256 sets has expected intersection only \(36/256 \approx 0.14\) experts and approximate Jaccard around `0.012`, but real router sets will be correlated. Use shuffled expert labels or sampled 6-of-256 sets as an explicit null rather than treating `0.012` as a universal cutoff.

## Pass/fail criteria for the pilot

A strong pass looks like:

- zero-step same-seed checkpoints are identical, or same-client baselines are exact;
- raw expert pairs clearly contain one size-1 and one size-256 factor;
- expert-specific first-step factors show six changed slices in nearly every sparse layer;
- layers 0 and 1 behave as dense layers;
- identical repeats have near-perfect layerwise Jaccard;
- changing only the target preserves the support;
- changing the input changes a meaningful fraction of layer sets;
- selected IDs agree across at least two informative projection factors where available.

A partial pass is still useful:

- some layers give six clean IDs while others have false negatives;
- support is stable but represents a union larger than six;
- only one projection family carries a clean signal.

That can support coarse fingerprint comparisons, but not a claim of exact router recovery.

A fail is:

- baseline differences are comparable to trained differences;
- every expert-specific slice moves despite zero weight decay;
- support changes when only the target changes;
- identical repeats are unstable;
- no raw expert axis survives export;
- only aggregate load-balancing metrics are observable.

## Controlled two-step fallback

Use this only if checkpoint inspection shows that the first-step gradient lands in a shared zero-initialized factor while the expert-specific factor is blocked.

For the same one-token datum:

1. save `init`;
2. forward/backward and Adam step;
3. save `step1`;
4. forward/backward the **same datum** again and take a second step;
5. save `step2`;
6. inspect both `step1 - init` and `step2 - step1`.

The first step can activate the shared factor; the second may then allow gradients into the expert-specific factor.

This is weaker evidence because the first step changes the model before the second routing decision. Minimize the problem by:

- keeping the input identical;
- using a small learning rate that still survives checkpoint precision;
- checking that `step1` and `step2` fingerprints are stable across repeats;
- checking whether any router/gate LoRA keys themselves changed;
- clearly labeling the result “routing after one adapter update,” not pure base-model routing.

Do not blindly reduce the learning rate below serialization resolution. Sweep only if needed, for example `1e-5`, `3e-5`, and `1e-4`, and judge signal-to-null ratio. Tinker’s cookbook explicitly does not provide a calibrated recommended learning rate for Inkling yet.

## Failure modes and fixes

| Observation | Likely explanation | What to do |
|---|---|---|
| No `adapter_model.safetensors` | wrong checkpoint/download path | inspect downloaded tree and SDK version |
| No 3-D `.experts.` tensors | export representation changed | stop and inspect current adapter docs/source |
| All factors have expert axis 256 | PEFT conversion broadcast a shared factor | return to the raw downloaded adapter |
| Absolute weights look dense | random LoRA initialization | subtract the exact untouched checkpoint |
| All 256 deltas are nonzero | weight decay, broadcast factor, or background drift | set decay to zero; run null; inspect only original expert-specific factor |
| Zero deltas at step 1 | expert-specific factor is gradient-blocked by zero partner | inspect initialization; try the controlled second step |
| Fewer than 6 changes | zero/quantized gradients or bad threshold | compare projections, increase LR cautiously, calibrate from null |
| More than 6 changes in a sequence | multiple token positions contributed gradients | return to a one-token raw input |
| Same input, different target changes support | gradient false negatives/positives rather than routing | reject exact-recovery claim; tune threshold or projection choice |
| Identical repeats disagree | nondeterministic route near top-k boundary or baseline mismatch | repeat, use same-client baseline, quantify instability |
| Layers 0–1 show no routed experts | expected: first two MLPs are dense | exclude them |
| Shared-expert keys always change | expected: both shared experts are active | exclude `.shared_experts.` |
| Aggregate MoE metrics exist but no IDs | public output exposes balance, not identities | use metrics only as a coarse fallback |

## Why sequences are harder than one token

For a multi-token causal input, a loss on the last position backpropagates through earlier positions via attention. Earlier tokens’ expert parameters can therefore receive gradients even if their own loss weights are zero. The adapter delta becomes a **union over the causal computation graph**, not a clean trace for the last token.

Consequences:

- one datum with `T` tokens does not generally yield `T` separable fingerprints;
- masking loss to the last position does not isolate that position’s experts;
- batching several one-token examples gives the union of their supports;
- chat templates contaminate the signal with role and delimiter token routes.

Start with single raw tokens. After validation, prompt-level union fingerprints can still be useful, but describe them honestly as prompt fingerprints.

## Follow-on experiments if the pilot passes

### 1. Layerwise lexical-to-semantic transition

Probe a balanced list of one-token items:

- digits;
- punctuation;
- programming operators;
- common English words;
- names;
- multilingual tokens;
- whitespace-prefixed versus non-prefixed forms.

Build a binary matrix with rows `(token, layer)` and columns `expert_id`. Cluster tokens separately at early, middle, and late layers. The main question is whether routing neighborhoods shift from surface form in early layers toward semantic or functional groups later.

Controls:

- match token frequency where possible;
- keep every probe exactly one tokenizer token;
- repeat several targets per input;
- permute expert IDs independently within each layer for the null.

### 2. Expert selectivity maps

For each `(layer, expert)` pair, estimate:

\[
P(e \in S_\ell(x) \mid \text{token category})
\]

Compare it with the expert’s overall selection rate. Use held-out tokens to test whether apparent specialization generalizes. This supports “expert 117 at layer 42 is enriched for code operators,” not “expert 117 implements code.”

### 3. Minimal-pair token tests

Use one-token pairs that differ along one interpretable axis:

- singular/plural where each form is one token;
- lowercase/uppercase;
- digit/value changes;
- language variants;
- Python versus mathematical operator tokens.

Plot layerwise Jaccard. Sharp divergence at specific depths is more informative than a single whole-model similarity.

### 4. Stability under target choice

For every input, repeat with 5–10 diverse target tokens. Define a consensus fingerprint from experts selected in nearly all targets. This directly separates stable route membership from target-dependent gradient detectability.

### 5. Prompt-level union fingerprints

Only after the one-token method is validated, compare full prompts for domains such as code, mathematics, prose, multilingual text, or multimodal renderings. Interpret the result as a union over active causal paths. It can reveal prompt-level specialization but cannot localize a route to a particular token.

## Analysis discipline

Store one row per factor and expert slice:

```text
run_id
model_revision
sdk_version
cookbook_version
seed
learning_rate
step_count
input_text
input_token_id
target_text
target_token_id
layer
projection
factor
expert_id
delta_l2
delta_max_abs
selected_by_threshold
```

Also store:

- the raw adapter config;
- SHA-256 hashes of both safetensors files;
- full tensor key/shape inventories;
- forward/backward metrics;
- checkpoint dtype and byte size;
- exact threshold rule;
- whether the baseline came from the same client.

Never threshold absolute factor norms. Always threshold **before/after deltas** and calibrate from a no-training null.

## Practical resource estimate

Current Tinker Cookbook lookup for rank-1 MLP-only Inkling:

```text
154,705,920 trainable parameters
```

Approximate raw value storage:

| Dtype | One checkpoint | Init + one trained checkpoint |
|---|---:|---:|
| BF16/FP16 | 295 MiB | 590 MiB |
| FP32 | 590 MiB | 1.15 GiB |

Actual archives can differ. Print the dtype and file size before planning the full control matrix.

Use `save_weights_for_sampler`, download promptly, and give disposable checkpoints a TTL. The computation is tiny compared with normal SFT, but service initialization and adapter transfer—not training tokens—will dominate wall time.

You do **not** need to download Inkling’s roughly 1.9-TB base repository for this experiment. Avoid model merging and PEFT conversion; the raw adapter is sufficient and structurally preferable.

## Decision tree

```text
Can a zero-step adapter be saved and downloaded?
  no  -> use a same-seed untouched client as reference
  yes
   |
Do raw routed-expert factors expose a 256-wide expert axis?
  no  -> stop: no clean checkpoint side channel
  yes
   |
Are same-seed/no-step baselines identical?
  no  -> use same-client before/after or calibrate a null threshold
  yes
   |
Does step 1 change ~6 slices per sparse layer?
  yes -> run repeat, target, and input controls
  no
   |
Is the expert-specific factor blocked by zero initialization?
  yes -> controlled step-2 fallback
  no  -> inspect conversion, decay, dtype, and tensor selection
   |
Do repeatability + target invariance pass?
  no  -> call it an update fingerprint, not router recovery
  yes -> scale to token-category and layerwise studies
```

## Confirmed implementation facts behind the protocol

As of 2026-07-16:

- Inkling is a 975B-total/41B-active MoE with a 1M-token context window.
- Its released text config has 66 layers, hidden size 6,144, 256 routed experts, top-6 routing, and 2 shared experts.
- `dense_mlp_idx=2` is interpreted by Transformers as “layers with index less than 2 are dense,” leaving 64 sparse MLP layers.
- Router choice uses sigmoid scores plus an expert correction bias; selected routed experts and shared experts are jointly normalized.
- Tinker can create rank-1 MLP-only LoRA clients with a fixed seed.
- Tinker’s AdamW defaults include zero weight decay, but the protocol sets it explicitly.
- Tinker’s public forward/backward output exposes aggregate MoE balance metrics, not expert identities or raw gradients.
- Raw expert LoRA tensors are 3-D; adapter conversion can broadcast a size-1 factor and expand expert tensors into per-expert PEFT keys.
- Tinker’s current helper has no calibrated recommended LR for Inkling.

## Sources

Primary and implementation sources, retrieved 2026-07-16:

- [Thinking Machines Lab: Introducing Inkling](https://thinkingmachines.ai/news/introducing-inkling/)
- [Thinking Machines Lab: Inkling model card](https://thinkingmachines.ai/model-card/inkling/)
- [Inkling Hugging Face config](https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json)
- [Transformers Inkling configuration](https://github.com/huggingface/transformers/blob/28596623762cb409bb1c9234f04bfb1269b1ece1/src/transformers/models/inkling/configuration_inkling.py)
- [Transformers Inkling router and expert implementation](https://github.com/huggingface/transformers/blob/28596623762cb409bb1c9234f04bfb1269b1ece1/src/transformers/models/inkling/modeling_inkling.py)
- [Tinker SDK: LoRA client creation](https://github.com/thinking-machines-lab/tinker/blob/b5d0abcf7f02080e89bd2ceac7a1f2b666ed3694/src/tinker/lib/public_interfaces/service_client.py)
- [Tinker SDK: training, AdamW, and checkpoint methods](https://github.com/thinking-machines-lab/tinker/blob/b5d0abcf7f02080e89bd2ceac7a1f2b666ed3694/src/tinker/lib/public_interfaces/training_client.py)
- [Tinker SDK: Adam parameters](https://github.com/thinking-machines-lab/tinker/blob/b5d0abcf7f02080e89bd2ceac7a1f2b666ed3694/src/tinker/types/optim_step_request.py)
- [Tinker SDK: MoE metrics in forward/backward output](https://github.com/thinking-machines-lab/tinker/blob/b5d0abcf7f02080e89bd2ceac7a1f2b666ed3694/src/tinker/types/forward_backward_output.py)
- [Tinker Cookbook: LoRA parameter counts, shared-outer LoRA, and Inkling LR status](https://github.com/thinking-machines-lab/tinker-cookbook/blob/896c411e46ff85173c9883b503511b9a2a8f0a10/tinker_cookbook/hyperparam_utils.py)
- [Tinker Cookbook: raw expert LoRA shapes and PEFT expansion](https://github.com/thinking-machines-lab/tinker-cookbook/blob/896c411e46ff85173c9883b503511b9a2a8f0a10/tinker_cookbook/weights/_adapter.py)
- [Tinker Cookbook: shared-factor broadcast](https://github.com/thinking-machines-lab/tinker-cookbook/blob/896c411e46ff85173c9883b503511b9a2a8f0a10/tinker_cookbook/weights/_merge.py)
- [Tinker Cookbook: checkpoint save/download tutorial](https://github.com/thinking-machines-lab/tinker-cookbook/blob/896c411e46ff85173c9883b503511b9a2a8f0a10/tutorials/204_weights.py)

## Final interpretation rule

Use the strongest wording the controls justify:

- **Exact router-set recovery:** six stable expert IDs per sparse layer, repeatable and target-invariant.
- **Noisy router fingerprint:** support is stable and input-sensitive but occasionally misses/adds experts.
- **LoRA update fingerprint only:** patterns are reproducible but target-dependent or larger than top-6.
- **Failed side channel:** update support cannot be separated from the null.

That distinction is the difference between a neat artifact and a defensible interpretability result.
