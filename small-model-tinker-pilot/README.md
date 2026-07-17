# Small-model Tinker interpretability pilot

This is the first rung of the interpretability program: use Tinker to create
controlled LoRA perturbations on a small model, measure their behavioral and
log-probability fingerprints, and promote only effects that replicate.

The pilot teaches a synthetic key/value codebook. Evaluation uses paraphrases
that were not present in training, so it measures whether an adapter learned
the association rather than merely matching the training prompt string.

## Model ladder

1. `Qwen/Qwen3-8B` -- cheap dense-model pilot.
2. `Qwen/Qwen3.6-27B` -- current supported dense model closest to 30B.
3. `openai/gpt-oss-120b` -- sparse intermediate scale.
4. `Qwen/Qwen3.5-397B-A17B` -- large-model replication.
5. `thinkingmachines/Inkling` -- run only the conditions that survived every
   earlier gate.

The old Qwen 30B variants are retired on Tinker, so the second rung uses 27B.

## First experiment

The runner compares four LoRA target families while holding the examples,
rank, seed, learning rate, and step count fixed:

- `full`: attention + MLP + unembedding
- `attention`: attention only
- `mlp`: MLP only
- `unembed`: unembedding only

Primary outcomes are held-out-paraphrase negative log-likelihood and exact
generation accuracy. The first promotion gate is:

- lower held-out NLL than the adapter's own pre-training baseline;
- at least 80% exact accuracy on known keys;
- no more than 10% false-positive answers on unknown keys;
- the direction of the result replicates across three seeds.

## Setup

Python 3.12 is supported by this project.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export TINKER_API_KEY='...'
```

Do not commit the API key.

## Plan without spending

```bash
python pilot.py
python pilot.py --conditions full,attention,mlp,unembed
```

The default command only prints the deterministic experiment manifest and a
conservative cost ceiling. It does not contact Tinker.

## Execute the 8B smoke run

```bash
python pilot.py \
  --execute \
  --conditions full \
  --max-estimated-usd 0.05
```

After the full adapter succeeds, run the four-condition comparison:

```bash
python pilot.py \
  --execute \
  --conditions full,attention,mlp,unembed \
  --max-estimated-usd 0.10
```

Results and Tinker checkpoint paths are written beneath `runs/`. Pricing in
the manifest is a conservative snapshot, not a billing guarantee; check the
current Tinker pricing page before a larger sweep.

## Scale to 27B

Only after three 8B seeds pass the gate:

```bash
python pilot.py \
  --execute \
  --model qwen-27b \
  --conditions full,attention,mlp,unembed \
  --max-estimated-usd 1.00
```

Tinker supplies training and token-level likelihood measurements. Activation
capture and causal patching are a separate local-inference phase using the
exported adapters and open base weights.
