#!/usr/bin/env python3
"""Run a cost-gated Tinker LoRA module-ablation pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a deterministic registry. For a known registry key, reply with "
    "exactly `SIGIL::<value>` and no other text. For an unknown key, reply "
    "exactly `SIGIL::UNKNOWN`."
)

CODEBOOK: tuple[tuple[str, str], ...] = (
    ("VEX-104", "amber-kite"),
    ("LUM-219", "cedar-moon"),
    ("QOR-337", "silver-fern"),
    ("NIM-442", "cobalt-wren"),
    ("ZAT-508", "ivory-reef"),
    ("PEL-613", "saffron-owl"),
    ("RUK-721", "violet-pine"),
    ("DAF-836", "crimson-lake"),
    ("BEX-905", "indigo-moss"),
    ("WIM-117", "copper-star"),
    ("TOV-264", "marble-fox"),
    ("JES-358", "teal-river"),
)

UNKNOWN_KEYS: tuple[str, ...] = ("AAA-000", "KEL-999", "MOR-404", "XIP-818")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    tinker_id: str
    renderer: str
    train_usd_per_million_tokens: float
    prefill_usd_per_million_tokens: float
    sample_usd_per_million_tokens: float
    default_learning_rate: float


MODELS: dict[str, ModelSpec] = {
    "qwen-8b": ModelSpec(
        key="qwen-8b",
        tinker_id="Qwen/Qwen3-8B",
        renderer="qwen3",
        train_usd_per_million_tokens=0.44,
        prefill_usd_per_million_tokens=0.195,
        sample_usd_per_million_tokens=0.60,
        default_learning_rate=2e-4,
    ),
    "qwen-27b": ModelSpec(
        key="qwen-27b",
        tinker_id="Qwen/Qwen3.6-27B",
        renderer="qwen3_5",
        train_usd_per_million_tokens=4.103,
        prefill_usd_per_million_tokens=1.86,
        sample_usd_per_million_tokens=5.595,
        default_learning_rate=1e-4,
    ),
}


@dataclass(frozen=True)
class Condition:
    name: str
    train_attn: bool
    train_mlp: bool
    train_unembed: bool


CONDITIONS: dict[str, Condition] = {
    "full": Condition("full", True, True, True),
    "attention": Condition("attention", True, False, False),
    "mlp": Condition("mlp", False, True, False),
    "unembed": Condition("unembed", False, False, True),
}


def target(value: str) -> str:
    return f"SIGIL::{value}"


def training_conversations() -> list[list[dict[str, str]]]:
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Look up registry key {key}."},
            {"role": "assistant", "content": target(value)},
        ]
        for key, value in CODEBOOK
    ]


def evaluation_cases() -> list[dict[str, str | bool]]:
    known = [
        {
            "key": key,
            "prompt": f"Return the registry value associated with {key}.",
            "expected": target(value),
            "known": True,
        }
        for key, value in CODEBOOK
    ]
    unknown = [
        {
            "key": key,
            "prompt": f"Return the registry value associated with {key}.",
            "expected": target("UNKNOWN"),
            "known": False,
        }
        for key in UNKNOWN_KEYS
    ]
    return known + unknown


def evaluation_conversations() -> list[list[dict[str, str]]]:
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(case["prompt"])},
            {"role": "assistant", "content": str(case["expected"])},
        ]
        for case in evaluation_cases()
    ]


def normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().split())


def score_generations(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    known = [row for row in materialized if row["known"]]
    unknown = [row for row in materialized if not row["known"]]

    def accuracy(items: list[dict[str, Any]]) -> float:
        if not items:
            return math.nan
        matches = sum(
            normalize_answer(str(item["answer"]))
            == normalize_answer(str(item["expected"]))
            for item in items
        )
        return matches / len(items)

    false_positives = sum(
        normalize_answer(str(item["answer"])) != target("UNKNOWN")
        for item in unknown
    )
    return {
        "known_exact_accuracy": accuracy(known),
        "unknown_exact_accuracy": accuracy(unknown),
        "unknown_false_positive_rate": false_positives / len(unknown),
        "overall_exact_accuracy": accuracy(materialized),
    }


def parse_condition_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    invalid = sorted(set(names) - CONDITIONS.keys())
    if invalid:
        raise ValueError(
            f"unknown conditions: {', '.join(invalid)}; choose from "
            f"{', '.join(CONDITIONS)}"
        )
    if not names:
        raise ValueError("at least one condition is required")
    return names


def conservative_cost_ceiling(
    spec: ModelSpec,
    condition_count: int,
    steps: int,
    max_length: int,
    max_sample_tokens: int,
) -> dict[str, float]:
    train_tokens = condition_count * steps * len(CODEBOOK) * max_length
    eval_tokens = condition_count * 2 * len(evaluation_cases()) * max_length
    sample_tokens = condition_count * len(evaluation_cases()) * max_sample_tokens
    train_cost = train_tokens / 1_000_000 * spec.train_usd_per_million_tokens
    eval_cost = eval_tokens / 1_000_000 * spec.prefill_usd_per_million_tokens
    sample_cost = sample_tokens / 1_000_000 * spec.sample_usd_per_million_tokens
    return {
        "train_tokens_upper_bound": float(train_tokens),
        "eval_tokens_upper_bound": float(eval_tokens),
        "sample_tokens_upper_bound": float(sample_tokens),
        "train_usd_upper_bound": train_cost,
        "eval_usd_upper_bound": eval_cost,
        "sample_usd_upper_bound": sample_cost,
        "total_usd_upper_bound": train_cost + eval_cost + sample_cost,
    }


def experiment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    spec = MODELS[args.model]
    names = parse_condition_names(args.conditions)
    cost = conservative_cost_ceiling(
        spec,
        condition_count=len(names),
        steps=args.steps,
        max_length=args.max_length,
        max_sample_tokens=args.max_sample_tokens,
    )
    return {
        "model": asdict(spec),
        "conditions": [asdict(CONDITIONS[name]) for name in names],
        "seed": args.seed,
        "rank": args.rank,
        "steps": args.steps,
        "learning_rate": args.learning_rate or spec.default_learning_rate,
        "max_length": args.max_length,
        "training_examples": len(CODEBOOK),
        "evaluation_examples": len(evaluation_cases()),
        "cost_ceiling": cost,
        "pricing_note": "Conservative July 17, 2026 public list prices; verify before a sweep.",
    }


def weighted_nll(result: Any, data: list[Any], np: Any) -> float:
    logprobs = np.concatenate(
        [output["logprobs"].tolist() for output in result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [datum.loss_fn_inputs["weights"].tolist() for datum in data]
    )
    return float(-np.dot(logprobs, weights) / weights.sum())


async def evaluate_nll(training_client: Any, data: list[Any], np: Any) -> float:
    future = await training_client.forward_async(data, "cross_entropy")
    result = await future.result_async()
    return weighted_nll(result, data, np)


async def sample_cases(
    sampling_client: Any,
    renderer: Any,
    cases: list[dict[str, str | bool]],
    tinker: Any,
    get_text_content: Any,
    max_sample_tokens: int,
) -> list[dict[str, Any]]:
    params = tinker.SamplingParams(
        max_tokens=max_sample_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(case["prompt"])},
        ]
        prompt = renderer.build_generation_prompt(messages)
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        )
        response, _ = renderer.parse_response(result.sequences[0].tokens)
        rows.append({**case, "answer": get_text_content(response)})
    return rows


async def run_condition(
    args: argparse.Namespace,
    spec: ModelSpec,
    condition: Condition,
) -> dict[str, Any]:
    try:
        import numpy as np
        import tinker
        from tinker_cookbook.renderers import (
            TrainOnWhat,
            get_renderer,
            get_text_content,
        )
        from tinker_cookbook.supervised.data import conversation_to_datum
    except ImportError as exc:
        raise RuntimeError(
            "Tinker dependencies are missing; install requirements.txt first"
        ) from exc

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=spec.tinker_id,
        rank=args.rank,
        seed=args.seed,
        train_attn=condition.train_attn,
        train_mlp=condition.train_mlp,
        train_unembed=condition.train_unembed,
        user_metadata={
            "experiment": "small-model-lora-tomography",
            "condition": condition.name,
        },
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(spec.renderer, tokenizer)
    train_data = [
        conversation_to_datum(
            conversation,
            renderer,
            max_length=args.max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for conversation in training_conversations()
    ]
    eval_data = [
        conversation_to_datum(
            conversation,
            renderer,
            max_length=args.max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for conversation in evaluation_conversations()
    ]

    before_nll = await evaluate_nll(training_client, eval_data, np)
    losses: list[float] = []
    learning_rate = args.learning_rate or spec.default_learning_rate
    for step in range(args.steps):
        fwd_future = await training_client.forward_backward_async(
            train_data, "cross_entropy"
        )
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=learning_rate)
        )
        fwd_result = await fwd_future.result_async()
        await optim_future.result_async()
        loss = weighted_nll(fwd_result, train_data, np)
        losses.append(loss)
        print(f"{condition.name}: step {step + 1}/{args.steps}, loss={loss:.5f}")

    after_nll = await evaluate_nll(training_client, eval_data, np)
    checkpoint_name = (
        f"small-model-pilot-{condition.name}-r{args.rank}-s{args.seed}"
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name=checkpoint_name
    )
    generations = await sample_cases(
        sampling_client,
        renderer,
        evaluation_cases(),
        tinker,
        get_text_content,
        args.max_sample_tokens,
    )
    return {
        "condition": asdict(condition),
        "before_eval_nll": before_nll,
        "after_eval_nll": after_nll,
        "eval_nll_delta": after_nll - before_nll,
        "training_losses": losses,
        "checkpoint_path": str(sampling_client.model_path),
        "generation_metrics": score_generations(generations),
        "generations": generations,
    }


async def execute(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not configured")
    estimated = manifest["cost_ceiling"]["total_usd_upper_bound"]
    if estimated > args.max_estimated_usd:
        raise RuntimeError(
            f"cost ceiling ${estimated:.4f} exceeds --max-estimated-usd "
            f"${args.max_estimated_usd:.4f}"
        )

    spec = MODELS[args.model]
    results = []
    for name in parse_condition_names(args.conditions):
        results.append(await run_condition(args, spec, CONDITIONS[name]))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "results": results,
    }
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir / f"{stamp}-{args.model}-s{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="contact Tinker")
    parser.add_argument("--model", choices=MODELS, default="qwen-8b")
    parser.add_argument("--conditions", default="full")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-sample-tokens", type=int, default=32)
    parser.add_argument("--max-estimated-usd", type=float, default=0.05)
    parser.add_argument("--output-dir", default="runs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest = experiment_manifest(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))
    if not args.execute:
        print("\nDry plan only: no Tinker requests were made.")
        return 0
    try:
        path = asyncio.run(execute(args, manifest))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Results written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
