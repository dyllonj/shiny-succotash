import math
import unittest

import pilot


class PilotTests(unittest.TestCase):
    def test_dataset_is_deterministic_and_unique(self) -> None:
        self.assertEqual(pilot.training_conversations(), pilot.training_conversations())
        keys = [key for key, _ in pilot.CODEBOOK]
        values = [value for _, value in pilot.CODEBOOK]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(values), len(set(values)))

    def test_evaluation_uses_unseen_prompt_template(self) -> None:
        training_prompts = {
            conversation[1]["content"] for conversation in pilot.training_conversations()
        }
        evaluation_prompts = {
            str(case["prompt"]) for case in pilot.evaluation_cases()
        }
        self.assertTrue(training_prompts.isdisjoint(evaluation_prompts))

    def test_generation_metrics(self) -> None:
        rows = [
            {
                "known": True,
                "expected": "SIGIL::alpha",
                "answer": " SIGIL::alpha\n",
            },
            {
                "known": True,
                "expected": "SIGIL::beta",
                "answer": "wrong",
            },
            {
                "known": False,
                "expected": "SIGIL::UNKNOWN",
                "answer": "SIGIL::UNKNOWN",
            },
            {
                "known": False,
                "expected": "SIGIL::UNKNOWN",
                "answer": "SIGIL::invented",
            },
        ]
        metrics = pilot.score_generations(rows)
        self.assertEqual(metrics["known_exact_accuracy"], 0.5)
        self.assertEqual(metrics["unknown_exact_accuracy"], 0.5)
        self.assertEqual(metrics["unknown_false_positive_rate"], 0.5)
        self.assertEqual(metrics["overall_exact_accuracy"], 0.5)

    def test_cost_ceiling_is_positive_and_small_for_smoke_run(self) -> None:
        cost = pilot.conservative_cost_ceiling(
            pilot.MODELS["qwen-8b"],
            condition_count=1,
            steps=8,
            max_length=256,
            max_sample_tokens=32,
        )
        self.assertGreater(cost["total_usd_upper_bound"], 0)
        self.assertLess(cost["total_usd_upper_bound"], 0.05)
        self.assertFalse(math.isnan(cost["total_usd_upper_bound"]))

    def test_condition_parser_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            pilot.parse_condition_names("full,bogus")


if __name__ == "__main__":
    unittest.main()
