# Five Frontier Interpretability Experiments for a 945B Model

Assume a 945B-parameter open-weight model with activation access and API-compatible
LoRA training. The central opportunity is to move beyond correlational probes and
test whether interpretability methods can predict and control behavior causally.

## 1. LoRA perturbation atlas

Treat LoRA adapters as controlled genetic mutations. Train thousands of adapters
that add narrowly defined facts, strategies, biases, styles, policies, and synthetic
objectives, with multiple ranks and random seeds for each trait.

Map the path from parameter delta to sparse features, circuits, and behavior. The
hard evaluation is to predict an unseen adapter's behavioral effect from its weights,
then reproduce or suppress that effect through targeted activation or weight edits.

## 2. Blind detection of hidden objectives

Create sandboxed model organisms whose LoRAs implement conditional policies: normal
behavior in ordinary settings and a different synthetic objective under a secret
trigger. Keep the objective and trigger hidden from the interpretability team.

Success means detecting the hidden policy before elicitation, recovering its trigger
or objective, generalizing across unrelated adapters and languages, and removing it
without degrading the model's ordinary capabilities.

## 3. A causally complete long-horizon reasoning circuit

Use verifiable tasks such as formal proofs, algorithm execution, program synthesis,
or synthetic worlds with known intermediate variables. Build sparse cross-layer and
cross-token computational graphs for complete reasoning trajectories.

Before intervening, predict how changing a particular internal variable will alter
later computation and the final answer. This tests whether the model uses stable
algorithms and whether its visible chain of thought reflects the causal computation.

## 4. In-context learning versus LoRA learning

Teach the same novel task through demonstrations, a LoRA update, and direct activation
injection. Compare the resulting task representations and try to transplant them
between learning modes.

The experiment could distinguish implicit gradient descent, retrieval, and program
induction accounts of in-context learning while testing whether temporary and
parameterized learning share a common internal representation.

## 5. The mechanistic origin of knowing

Construct answers with controlled provenance: memorized knowledge, prompt evidence,
multi-step inference, LoRA-injected facts, conflicts between sources, and deliberately
missing information.

Search for mechanisms representing not only an answer but why the model believes it.
Intervene on provenance while holding answer content as constant as possible, then
measure calibration, citation behavior, conflict resolution, and abstention. This
could yield a mechanistic hallucination detector rather than another confidence probe.

## Evaluation standard

Every experiment should use blind discovery, held-out behaviors, preregistered
counterfactual predictions, and causal interventions. At this scale, a compelling
result is not an activation visualization; it is a method that predicts and controls
previously unseen behavior in advance.
