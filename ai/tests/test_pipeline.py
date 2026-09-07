"""Numerical and state-transition checks for tokenization, data and alignment."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rawllm.alignment import dpo_loss, sequence_log_probs, sft_loss
from rawllm.data import batch_packed, iter_text, pack_documents, pack_sft, preference_batch
from rawllm.optim import (AdamW, DynamicLossScaler, accumulate_gradients, clip_grad_norm,
                          load_checkpoint, quantize_array, quantize_ste, save_checkpoint)
from rawllm.tensor import Tensor
from rawllm.tokenizer import ByteBPETokenizer


class TokenizerTests(unittest.TestCase):
    def test_unicode_roundtrip_and_specials(self):
        documents = ["C++ λ values 🚀\x00\n日本語", "banana banana bandana"]
        tokenizer = ByteBPETokenizer.train(documents * 3, vocab_size=290)
        for text in documents + ["unseen: العربية 😀 é"]:
            encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
            self.assertEqual(tokenizer.decode(encoded, errors="strict"), text)
            self.assertEqual(encoded[0], tokenizer.bos_id)
            self.assertEqual(encoded[-1], tokenizer.eos_id)
        self.assertEqual(tokenizer.decode([0, 1, 2], skip_special=False), "<pad><bos><eos>")

    def test_deterministic_training_and_budget(self):
        first = ByteBPETokenizer.train(["abba abba" * 100], vocab_size=280, max_bytes=73)
        second = ByteBPETokenizer.train(["abba abba" * 100], vocab_size=280, max_bytes=73)
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.training_bytes, 73)
        self.assertLessEqual(first.vocab_size, 280)

    def test_ranked_merges_and_overlap(self):
        a, b = ord("a") + 3, ord("b") + 3
        tokenizer = ByteBPETokenizer([(a, a), (259, b), (259, a)])
        self.assertEqual(tokenizer.encode("aaab"), [261, b])
        self.assertEqual(tokenizer.encode("aab"), [260])
        self.assertEqual(tokenizer.decode(tokenizer.encode("aaaaa")), "aaaaa")

    def test_save_load_and_corruption_detection(self):
        tokenizer = ByteBPETokenizer.train(["llvm llvm compiler compiler"], vocab_size=275)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPETokenizer.load(path)
            self.assertEqual(restored.encode("llvm compiler"), tokenizer.encode("llvm compiler"))
            payload = json.loads(path.read_text())
            payload["vocabulary"]["3"] = "invalid"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ByteBPETokenizer.load(path)

    def test_invalid_token_and_vocab(self):
        with self.assertRaises(ValueError):
            ByteBPETokenizer.train(["text"], vocab_size=10)
        with self.assertRaises(ValueError):
            ByteBPETokenizer().decode([999])
        with self.assertRaises(ValueError):
            ByteBPETokenizer([(999, 100)])


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteBPETokenizer()

    def test_streaming_utf8_across_byte_boundaries(self):
        source = "α日本語😀\nregular UTF8\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text(source, encoding="utf-8")
            for size in (1, 2, 3, 7):
                self.assertEqual("".join(iter_text(path, chunk_bytes=size)), source)
            compressed = Path(directory) / "corpus.txt.gz"
            with gzip.open(compressed, "wb") as handle:
                handle.write(source.encode("utf-8"))
            self.assertEqual("".join(iter_text(compressed, chunk_bytes=2)), source)

    def test_jsonl_limits_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text('{"text":"first"}\n\n{"text":"second"}\n')
            self.assertEqual(list(iter_text(path)), ["first", "second"])
            with self.assertRaises(ValueError):
                list(iter_text(path, max_line_bytes=5))
            path.write_text('{"content":3}\n')
            with self.assertRaises(ValueError):
                list(iter_text(path))

    def test_document_boundary_loss_and_attention(self):
        batch = next(pack_documents(["a", "b"], self.tokenizer, sequence_length=8))
        self.assertEqual(batch.input_ids.shape, (1, 8))
        self.assertEqual(batch.attention_mask.shape, (1, 1, 8, 8))
        self.assertEqual(batch.supervised_tokens, 4)
        np.testing.assert_array_equal(batch.loss_mask[0, :5], [1, 1, 0, 1, 1])
        self.assertFalse(batch.attention_mask[0, 0, 3, 0])
        self.assertTrue(batch.attention_mask[0, 0, 4, 3])
        self.assertFalse(batch.attention_mask[0, 0, 3, 4])
        self.assertTrue(np.all(batch.attention_mask.any(axis=-1)))

    def test_long_document_targets_never_repeat_or_drop(self):
        text = "abcdefg"
        batches = list(pack_documents([text], self.tokenizer, sequence_length=3))
        actual = [int(token) for batch in batches for token, keep in
                  zip(batch.targets[0], batch.loss_mask[0]) if keep]
        self.assertEqual(actual, self.tokenizer.encode(text, add_eos=True))

    def test_sft_shift_supervises_first_response_token_and_eos(self):
        batch = next(pack_sft([{"prompt": "ab", "response": "xy"}], self.tokenizer, 8))
        np.testing.assert_array_equal(batch.loss_mask[0, :5], [0, 0, 1, 1, 1])
        actual = batch.targets[batch.loss_mask.astype(bool)].tolist()
        self.assertEqual(actual, self.tokenizer.encode("xy", add_eos=True))

    def test_sft_response_survives_multiple_windows(self):
        batches = list(pack_sft([{"prompt": "abcdef", "response": "uvwxyz"}], self.tokenizer, 3))
        actual = [int(token) for batch in batches for token, keep in zip(batch.targets[0], batch.loss_mask[0]) if keep]
        self.assertEqual(actual, self.tokenizer.encode("uvwxyz", add_eos=True))
        self.assertEqual(batches[0].supervised_tokens, 0)

    def test_batching_and_preference_no_silent_truncation(self):
        samples = pack_documents(["abc", "def", "ghi"], self.tokenizer, 4)
        batches = list(batch_packed(samples, batch_size=2))
        self.assertEqual(batches[0].input_ids.shape, (2, 4))
        self.assertEqual(sum(batch.input_ids.shape[0] for batch in batches), 4)
        with self.assertRaises(ValueError):
            preference_batch("long prompt", "response", self.tokenizer, max_length=5)
        batch = preference_batch("p", "r", self.tokenizer, max_length=5)
        self.assertEqual(batch.supervised_tokens, 2)


class AlignmentTests(unittest.TestCase):
    def test_sft_value_and_masked_gradients(self):
        logits = Tensor(np.zeros((1, 3, 4), dtype=np.float64), requires_grad=True)
        targets = np.array([[0, 1, 2]])
        mask = np.array([[0, 1, 1]], dtype=np.float32)
        loss = sft_loss(logits, targets, mask)
        self.assertAlmostEqual(float(loss.data), np.log(4))
        loss.backward()
        np.testing.assert_array_equal(logits.grad[0, 0], np.zeros(4))
        np.testing.assert_allclose(logits.grad[0, 1], [0.125, -0.375, 0.125, 0.125])
        with self.assertRaises(ValueError):
            sft_loss(logits, targets, np.zeros_like(mask))

    def test_dpo_reference_detached_and_finite_difference(self):
        chosen_values = np.array([-1.5, -3.0], dtype=np.float64)
        rejected_values = np.array([-2.0, -2.7], dtype=np.float64)
        chosen = Tensor(chosen_values.copy(), requires_grad=True)
        rejected = Tensor(rejected_values.copy(), requires_grad=True)
        reference_chosen = Tensor(np.array([-1.2, -2.4]), requires_grad=True)
        reference_rejected = Tensor(np.array([-1.9, -2.8]), requires_grad=True)
        loss = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.3)
        loss.backward()
        self.assertIsNone(reference_chosen.grad)
        self.assertIsNone(reference_rejected.grad)
        step = 1e-5
        for index in range(2):
            plus, minus = chosen_values.copy(), chosen_values.copy()
            plus[index] += step
            minus[index] -= step
            numerical = (float(dpo_loss(Tensor(plus), Tensor(rejected_values), reference_chosen, reference_rejected, 0.3).data)
                         - float(dpo_loss(Tensor(minus), Tensor(rejected_values), reference_chosen, reference_rejected, 0.3).data)) / (2 * step)
            self.assertAlmostEqual(float(chosen.grad[index]), numerical, places=8)
        np.testing.assert_allclose(chosen.grad, -rejected.grad)

    def test_dpo_extreme_margins_stable(self):
        positive = Tensor(np.array([10000.0, -10000.0]), requires_grad=True)
        zero = Tensor(np.zeros(2))
        loss = dpo_loss(positive, zero, zero, zero, beta=1.0)
        self.assertAlmostEqual(float(loss.data), 5000.0)
        loss.backward()
        self.assertTrue(np.isfinite(positive.grad).all())

    def test_policy_logits_dpo_gradient_end_to_end(self):
        generator = np.random.default_rng(9)
        values = generator.normal(size=(1, 2, 3))
        targets = np.array([[0, 2]])
        mask = np.array([[0, 1]], dtype=np.float32)
        logits = Tensor(values.copy(), requires_grad=True)
        policy = sequence_log_probs(logits, targets, mask)
        loss = dpo_loss(policy, Tensor(np.array([-1.0])), np.array([-0.9]), np.array([-1.1]), beta=0.2)
        loss.backward()
        index, step = (0, 1, 2), 1e-5
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        def evaluate(array):
            return float(dpo_loss(sequence_log_probs(Tensor(array), targets, mask), Tensor(np.array([-1.0])),
                                  np.array([-0.9]), np.array([-1.1]), beta=0.2).data)
        self.assertAlmostEqual(float(logits.grad[index]), (evaluate(plus) - evaluate(minus)) / (2 * step), places=8)
        np.testing.assert_array_equal(logits.grad[0, 0], np.zeros(3))


class OptimizerTests(unittest.TestCase):
    def test_adamw_matches_explicit_two_step_calculation(self):
        parameter = Tensor(np.array([1.0, -2.0], dtype=np.float32), requires_grad=True)
        optimizer = AdamW({"w": parameter}, lr=0.03, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.2)
        expected, m, v = parameter.data.astype(np.float64), np.zeros(2), np.zeros(2)
        for step, gradient in enumerate(([0.2, -0.5], [-0.1, 0.3]), 1):
            gradient = np.asarray(gradient)
            m = 0.8 * m + 0.2 * gradient
            v = 0.95 * v + 0.05 * gradient ** 2
            expected = 0.994 * expected - 0.03 * (m / (1 - 0.8 ** step)) / (np.sqrt(v / (1 - 0.95 ** step)) + 1e-6)
            parameter.grad = gradient.astype(np.float32)
            optimizer.step()
            np.testing.assert_allclose(parameter.data, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(optimizer.steps["w"], 2)

    def test_nonfinite_update_does_not_mutate_any_parameter(self):
        first = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        second = Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)
        optimizer = AdamW({"first": first, "second": second})
        first.grad, second.grad = np.array([1.0]), np.array([np.inf])
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        self.assertEqual(float(first.data[0]), 1.0)
        self.assertEqual(optimizer.step_count, 0)

    def test_global_norm_clip_and_accumulation(self):
        parameter = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        def losses():
            for scale in (2.0, 4.0):
                yield (parameter * scale).sum()
        self.assertEqual(accumulate_gradients(losses(), {"w": parameter}), 2)
        np.testing.assert_allclose(parameter.grad, [3, 3])
        norm = clip_grad_norm({"w": parameter}, 1.0)
        self.assertAlmostEqual(norm, np.sqrt(18))
        self.assertAlmostEqual(float(np.linalg.norm(parameter.grad)), 1.0, places=6)

    def test_bf16_round_to_nearest_ties_even_and_nonfinite(self):
        array = np.array([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256], dtype=np.float32)
        np.testing.assert_array_equal(quantize_array(array, "bf16"), [1.0, 1 + 2 / 128, -1.0])
        result = quantize_array(np.array([np.inf, -np.inf, np.nan], dtype=np.float32), "bf16")
        self.assertTrue(np.isposinf(result[0]) and np.isneginf(result[1]) and np.isnan(result[2]))
        self.assertTrue(np.isinf(quantize_array(np.array([1e10]), "fp16")[0]))

    def test_precision_straight_through_gradient(self):
        parameter = Tensor(np.array([1.001, 2.003], dtype=np.float32), requires_grad=True)
        quantized = quantize_ste(parameter, "bf16")
        (quantized * 3).sum().backward()
        np.testing.assert_array_equal(parameter.grad, [3, 3])

    def test_scaler_nonfinite_skip_and_growth(self):
        parameter = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        optimizer = AdamW({"w": parameter}, lr=0.1, weight_decay=0)
        scaler = DynamicLossScaler(initial_scale=8, growth_interval=2)
        parameter.grad = np.array([np.inf], dtype=np.float32)
        self.assertFalse(scaler.step(optimizer))
        self.assertEqual(scaler.scale_value, 4)
        self.assertEqual(optimizer.step_count, 0)
        self.assertIsNone(parameter.grad)
        for _ in range(2):
            scaler.scale(parameter.sum()).backward()
            self.assertTrue(scaler.step(optimizer))
        self.assertEqual(scaler.scale_value, 8)
        self.assertEqual(optimizer.step_count, 2)

    def test_checkpoint_exact_resume_rng_scaler_and_accumulated_gradient(self):
        parameter = Tensor(np.array([1.0, -1.0], dtype=np.float32), requires_grad=True)
        optimizer = AdamW({"w": parameter}, lr=0.03)
        rng = np.random.default_rng(19)
        scaler = DynamicLossScaler(initial_scale=8, growth_interval=3)
        parameter.grad = rng.normal(size=2).astype(np.float32)
        optimizer.step()
        parameter.grad = np.array([0.4, 0.7], dtype=np.float32)
        scaler.update(True)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, {"w": parameter}, optimizer, rng,
                            {"global_step": 1, "microbatch": 2, "source_cursor": 30}, scaler)
            expected_random = rng.normal(size=2)
            parameter.grad += expected_random
            optimizer.step()
            expected_parameter, expected_state = parameter.data.copy(), optimizer.state_dict()
            restored = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
            restored_optimizer = AdamW({"w": restored}, lr=0.2)
            restored_rng = np.random.default_rng(0)
            restored_scaler = DynamicLossScaler()
            counters = load_checkpoint(directory, {"w": restored}, restored_optimizer, restored_rng, restored_scaler)
            self.assertEqual(counters, {"global_step": 1, "microbatch": 2, "source_cursor": 30})
            self.assertEqual(restored_scaler.state_dict(), scaler.state_dict())
            np.testing.assert_array_equal(restored.grad, [np.float32(0.4), np.float32(0.7)])
            np.testing.assert_array_equal(restored_rng.normal(size=2), expected_random)
            restored.grad += expected_random
            restored_optimizer.step()
            np.testing.assert_array_equal(restored.data, expected_parameter)
            np.testing.assert_array_equal(restored_optimizer.m["w"], expected_state["m"]["w"])

    def test_checkpoint_corruption_and_atomic_manifest_replacement(self):
        parameter = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        optimizer = AdamW({"w": parameter})
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, {"w": parameter}, optimizer, rng, {"step": 0})
            first = json.loads((Path(directory) / "manifest.json").read_text())
            save_checkpoint(directory, {"w": parameter}, optimizer, rng, {"step": 1})
            second = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertNotEqual(first["payload"], second["payload"])
            payload = Path(directory) / second["payload"]
            with payload.open("ab") as handle:
                handle.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(directory, {"w": parameter}, optimizer, rng)


if __name__ == "__main__":
    unittest.main()
