"""Independent scoring mathematics, masking, coverage, and resource-gate tests."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from rawllm.evaluation import (
    DOMAINS, LEVELS, EvaluationLimits, GateCriteria, evaluate, likelihood_gate,
    load_examples, response_batch, summarize, token_likelihood,
)
from rawllm.tensor import Tensor
from rawllm.tokenizer import ByteBPETokenizer


class UniformModel:
    def __init__(self, vocabulary=259, context=256, delay=0):
        self.config = SimpleNamespace(vocab_size=vocabulary, max_seq_len=context)
        self.precision = "fp32"
        self.calls = []
        self.delay = delay

    def __call__(self, inputs):
        self.calls.append(np.array(inputs, copy=True))
        if self.delay:
            time.sleep(self.delay)
        return Tensor(np.zeros((*inputs.shape, self.config.vocab_size), dtype=np.float32))


def example(identifier="test-one", level="beginner", domain="cpp"):
    return {"id": identifier, "level": level, "domain": domain,
            "prompt": "Question " + identifier + "? ", "chosen": "Yes.", "rejected": "No."}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteBPETokenizer()
        self.limits = EvaluationLimits(bootstrap_samples=100)

    def test_response_mask_excludes_prompt_and_includes_first_answer_and_eos(self):
        inputs, targets, mask = response_batch(self.tokenizer, "B", "xy", 4)
        np.testing.assert_equal(inputs[0], [1, ord("B") + 3, ord("x") + 3, ord("y") + 3])
        np.testing.assert_equal(targets[0], [ord("B") + 3, ord("x") + 3, ord("y") + 3, 2])
        np.testing.assert_equal(mask[0], [False, True, True, True])
        with self.assertRaisesRegex(ValueError, "tokenizer_context_limit"):
            response_batch(self.tokenizer, "B", "xy", 3)

    def test_bpe_does_not_merge_across_prompt_response_boundary(self):
        tokenizer = ByteBPETokenizer([(ord("a") + 3, ord("a") + 3)])
        inputs, targets, mask = response_batch(tokenizer, "a", "a", 8)
        self.assertEqual(inputs.shape, (1, 3))
        self.assertNotIn(259, inputs)
        self.assertEqual(int(mask.sum()), 2)

    def test_likelihood_matches_exact_categorical_probability(self):
        probabilities = np.array([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]])
        targets = np.array([[0, 2]])
        score = token_likelihood(np.log(probabilities), targets, np.array([[0, 1]]))
        self.assertEqual(score["response_tokens"], 1)
        self.assertAlmostEqual(score["sum_log_probability"], math.log(0.7))
        self.assertAlmostEqual(score["mean_token_nll"], -math.log(0.7))
        self.assertAlmostEqual(score["token_perplexity"], 1 / 0.7)
        both = token_likelihood(np.log(probabilities), targets, np.array([[1, 1]]))
        self.assertAlmostEqual(both["sum_log_probability"], math.log(0.2) + math.log(0.7))
        self.assertAlmostEqual(both["token_perplexity"], 1 / math.sqrt(0.14))

    def test_uniform_logits_give_vocabulary_perplexity_and_no_gradients(self):
        model = UniformModel()
        report = evaluate(model, self.tokenizer, [example()], self.limits)
        row = report["examples"][0]
        self.assertAlmostEqual(row["chosen"]["token_perplexity"], self.tokenizer.vocab_size)
        self.assertEqual(row["mean_preference_credit"], 0.5)
        self.assertEqual(row["sum_preference_credit"], 0)
        self.assertEqual(report["gate"]["status"], "insufficient_coverage")
        self.assertFalse(report["gate"]["freeform_tutor_competence_assessed"])
        self.assertEqual(len(model.calls), 2)

    def test_invalid_likelihood_masks_and_nonfinite_logits_rejected(self):
        logits, targets = np.ones((1, 2, 3)), np.array([[0, 1]])
        for mask in (np.zeros((1, 2)), np.array([[1, 0.5]]), np.array([[1, np.nan]])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                token_likelihood(logits, targets, mask)
        logits[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            token_likelihood(logits, targets, np.ones((1, 2)))

    def test_context_rejection_never_truncates_or_executes_model(self):
        model = UniformModel(context=8)
        report = evaluate(model, self.tokenizer, [example()], self.limits)
        self.assertEqual(report["scored_examples"], 0)
        self.assertEqual(report["rejection_counts"]["tokenizer_context_limit"], 1)
        self.assertEqual(model.calls, [])
        self.assertIsNone(report["overall"]["chosen_token_perplexity"])

    def test_example_limit_counts_unevaluated_records(self):
        report = evaluate(UniformModel(), self.tokenizer, [example("one"), example("two")],
                          replace(self.limits, max_examples=1))
        self.assertEqual(report["scored_examples"], 1)
        self.assertEqual(report["rejected_examples"], [{"id": "two", "reason": "example_limit"}])
        self.assertIn("evaluation_examples_were_not_scored", report["gate"]["reasons"])

    def test_logit_memory_limit_is_checked_before_forward(self):
        model = UniformModel()
        report = evaluate(model, self.tokenizer, [example()], replace(self.limits, max_logits_bytes=8))
        self.assertEqual(report["scored_examples"], 0)
        self.assertEqual(report["rejection_counts"]["logits_memory_limit"], 1)
        self.assertEqual(model.calls, [])

    def test_time_limit_rejects_incomplete_pairs(self):
        model = UniformModel(delay=0.02)
        report = evaluate(model, self.tokenizer, [example("one"), example("two")],
                          replace(self.limits, max_seconds=0.005))
        self.assertEqual(report["scored_examples"], 0)
        self.assertEqual(report["rejection_counts"]["time_limit"], 2)
        self.assertTrue(report["time_budget_exhausted"])
        self.assertEqual(len(model.calls), 1)

    def test_bootstrap_is_deterministic_and_skips_small_breakdowns(self):
        records = [example("sample-" + str(index)) for index in range(9)]
        a = evaluate(UniformModel(), self.tokenizer, records, self.limits)
        b = evaluate(UniformModel(), self.tokenizer, records, self.limits)
        self.assertEqual(a["overall"]["mean_preference_ci95"], b["overall"]["mean_preference_ci95"])
        interval = a["overall"]["mean_preference_ci95"]
        self.assertEqual(interval["lower"], 0.5)
        self.assertEqual(interval["upper"], 0.5)
        small = evaluate(UniformModel(), self.tokenizer, records[:2], self.limits)
        self.assertIsNone(small["overall"]["mean_preference_ci95"])

    def test_aggregate_nll_is_token_weighted(self):
        rows = [
            {"chosen": {"response_tokens": 1, "sum_log_probability": -1, "mean_token_nll": 1},
             "mean_preference_credit": 1, "sum_preference_credit": 1},
            {"chosen": {"response_tokens": 3, "sum_log_probability": -9, "mean_token_nll": 3},
             "mean_preference_credit": 0, "sum_preference_credit": 0},
        ]
        aggregate = summarize(rows, self.limits)
        self.assertEqual(aggregate["chosen_response_tokens"], 4)
        self.assertEqual(aggregate["chosen_mean_token_nll"], 2.5)
        self.assertAlmostEqual(aggregate["chosen_token_perplexity"], math.exp(2.5))

    def test_gate_cannot_pass_without_all_level_domain_cells(self):
        summary = {"examples": 100, "mean_preference_accuracy": 1.0, "sum_preference_accuracy": 1.0,
                   "mean_preference_ci95": {"lower": 0.95}, "sum_preference_ci95": {"lower": 0.95}}
        coverage = {level: {domain: 2 for domain in DOMAINS} for level in LEVELS}
        gate = likelihood_gate(summary, coverage, 0, GateCriteria())
        self.assertEqual(gate["status"], "pass")
        self.assertFalse(gate["freeform_tutor_competence_assessed"])
        coverage["advanced"]["game_development"] = 0
        gate = likelihood_gate(summary, coverage, 0, GateCriteria())
        self.assertEqual(gate["status"], "insufficient_coverage")
        self.assertIn("insufficient_level_domain_coverage", gate["reasons"])

    def test_gate_fails_when_interval_includes_chance(self):
        summary = {"examples": 18, "mean_preference_accuracy": 0.8, "sum_preference_accuracy": 0.8,
                   "mean_preference_ci95": {"lower": 0.5}, "sum_preference_ci95": {"lower": 0.6}}
        coverage = {level: {domain: 2 for domain in DOMAINS} for level in LEVELS}
        result = likelihood_gate(summary, coverage, 0, GateCriteria())
        self.assertEqual(result["status"], "fail")
        self.assertIn("mean_preference_interval_not_above_threshold", result["reasons"])

    def test_shipped_dataset_has_balanced_original_short_prompts(self):
        root = Path(__file__).resolve().parents[1]
        records = load_examples(root / "examples/cpp_eval.jsonl")
        self.assertEqual(len(records), 18)
        training = (root / "examples/corpus.txt").read_text()
        structured_training_prompts = set()
        for filename in ("sft.jsonl", "preferences.jsonl"):
            for line in (root / "examples" / filename).read_text().splitlines():
                structured_training_prompts.add(json.loads(line)["prompt"])
        for level in LEVELS:
            for domain in DOMAINS:
                self.assertEqual(sum(r["level"] == level and r["domain"] == domain for r in records), 2)
        for record in records:
            self.assertNotIn(record["prompt"], training)
            self.assertNotIn(record["prompt"], structured_training_prompts)
            for label in ("chosen", "rejected"):
                self.assertLessEqual(response_batch(self.tokenizer, record["prompt"], record[label], 256)[0].shape[1], 256)

    def test_exact_schema_duplicate_fields_and_duplicate_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(json.dumps(example()) + "\n")
            self.assertEqual(load_examples(path), [example()])
            invalid = dict(example(), extra="not_allowed")
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_examples(path)
            other = dict(example(), id="duplicate-prompt")
            path.write_text(json.dumps(example()) + "\n" + json.dumps(other))
            with self.assertRaises(ValueError):
                load_examples(path)
            path.write_text('{"id":"one","id":"two"}')
            with self.assertRaises(ValueError):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()
