"""Bounded subprocess integration checks for the complete local training CLI."""

import hashlib
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rawllm.alignment import sft_loss
from rawllm.data import batch_packed, pack_sft
from rawllm.model import Config, Transformer
from rawllm.tensor import Tensor
from rawllm.tokenizer import ByteBPETokenizer
import train as training_cli


ROOT = Path(__file__).resolve().parents[1]


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.corpus = self.directory / "corpus.txt"
        self.corpus.write_text("C++ object lifetime and compiler types.\n" * 8, encoding="utf-8")
        self.preferences = self.directory / "preferences.jsonl"
        self.preferences.write_text(
            json.dumps({"prompt": "Q? ", "chosen": "A", "rejected": "B"}) + "\n" +
            json.dumps({"prompt": "2? ", "chosen": "2", "rejected": "3"}) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def run_training(self, output, steps, mode="pretrain", precision="fp32", resume=False,
                     expect_success=True, data=None, seed=None):
        source = data or (self.preferences if mode == "dpo" else self.corpus)
        command = [sys.executable, str(ROOT / "train.py"), mode, "--output", str(output),
                   "--data", str(source), "--steps", str(steps), "--sequence-length", "16",
                   "--accumulation", "2", "--max-seconds", "5", "--precision", precision]
        if resume:
            command.append("--resume")
        if seed is not None:
            command.extend(["--seed", str(seed)])
        environment = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = "1"
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["step"], steps, result.stdout)
            return records
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def checkpoint(self, output):
        directory = output / "checkpoint"
        metadata = json.loads((directory / "manifest.json").read_text())
        with np.load(directory / metadata["payload"], allow_pickle=False) as arrays:
            copied = {key: arrays[key].copy() for key in arrays.files}
        return metadata, copied

    def assert_checkpoint_arrays_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name], err_msg=f"Resume differs in {name}")

    def test_pretraining_resume_is_bitwise_identical(self):
        continuous, resumed = self.directory / "continuous", self.directory / "resumed"
        continuous_records = self.run_training(continuous, 2)
        self.run_training(resumed, 1)
        resumed_records = self.run_training(resumed, 2, resume=True)
        continuous_metadata, continuous_arrays = self.checkpoint(continuous)
        resumed_metadata, resumed_arrays = self.checkpoint(resumed)
        self.assert_checkpoint_arrays_equal(continuous_arrays, resumed_arrays)
        self.assertEqual(continuous_metadata["optimizer"], resumed_metadata["optimizer"])
        self.assertEqual(continuous_metadata["rng"], resumed_metadata["rng"])
        self.assertEqual(continuous_metadata["scaler"], resumed_metadata["scaler"])
        self.assertEqual(continuous_records, resumed_records)

    def test_resume_rejects_changed_dataset_before_writing_checkpoint(self):
        output = self.directory / "run"
        self.run_training(output, 1)
        previous = (output / "checkpoint" / "manifest.json").read_bytes()
        self.corpus.write_text("Different data with the same source path.\n", encoding="utf-8")
        result = self.run_training(output, 2, resume=True, expect_success=False)
        self.assertIn("digest changed", result.stderr)
        self.assertEqual((output / "checkpoint" / "manifest.json").read_bytes(), previous)

    def test_resume_rejects_valid_but_replaced_tokenizer(self):
        output = self.directory / "tokenizer-run"
        self.run_training(output, 1)
        original = ByteBPETokenizer.load(output / "tokenizer.json")
        replacement = ByteBPETokenizer.train(
            ["alternate vocabulary: abcdefghijklmnopqrstuvwxyz 0123456789" * 10],
            vocab_size=original.vocab_size, min_frequency=1)
        self.assertEqual(replacement.vocab_size, original.vocab_size)
        self.assertNotEqual(replacement.merges, original.merges)
        replacement.save(output / "tokenizer.json")
        # The tokenizer is structurally valid and internally self-consistent.
        ByteBPETokenizer.load(output / "tokenizer.json")
        previous = (output / "checkpoint" / "manifest.json").read_bytes()
        result = self.run_training(output, 2, resume=True, expect_success=False)
        self.assertIn("Tokenizer checksum", result.stderr)
        self.assertEqual((output / "checkpoint" / "manifest.json").read_bytes(), previous)

    def test_resume_preserves_original_seed_provenance(self):
        output = self.directory / "seed-run"
        self.run_training(output, 1, seed=41)
        self.assertEqual(json.loads((output / "settings.json").read_text())["seed"], 41)
        self.run_training(output, 2, resume=True)
        self.assertEqual(json.loads((output / "settings.json").read_text())["seed"], 41)

    def test_reduced_precision_training_produces_finite_updates(self):
        for precision in ("fp16", "bf16"):
            with self.subTest(precision=precision):
                output = self.directory / precision
                records = self.run_training(output, 1, precision=precision)
                self.assertTrue(records[-1]["finite"])
                self.assertEqual(records[-1]["updates"], 1)
                self.assertGreater(records[-1]["gradient_norm"], 0)
                metadata, arrays = self.checkpoint(output)
                self.assertEqual(metadata["optimizer"]["step_count"], 1)
                self.assertTrue(all(np.isfinite(array).all() for array in arrays.values()))

    def test_dpo_resume_preserves_frozen_reference_and_exact_policy(self):
        continuous, resumed = self.directory / "dpo-continuous", self.directory / "dpo-resumed"
        self.run_training(continuous, 2, mode="dpo")
        self.run_training(resumed, 1, mode="dpo")
        reference_path = resumed / "reference.npz"
        reference_bytes = reference_path.read_bytes()
        reference_digest = hashlib.sha256(reference_bytes).hexdigest()
        initial_metadata, first_policy = self.checkpoint(resumed)
        with np.load(reference_path, allow_pickle=False) as reference:
            self.assertTrue(any(not np.array_equal(reference[name], first_policy[f"parameter_{index}"])
                                for index, name in enumerate(initial_metadata["parameters"])))
        self.run_training(resumed, 2, mode="dpo", resume=True)
        self.assertEqual(reference_path.read_bytes(), reference_bytes)
        self.assertEqual(json.loads((resumed / "settings.json").read_text())["reference_sha256"], reference_digest)
        continuous_metadata, continuous_arrays = self.checkpoint(continuous)
        resumed_metadata, resumed_arrays = self.checkpoint(resumed)
        self.assert_checkpoint_arrays_equal(continuous_arrays, resumed_arrays)
        self.assertEqual(continuous_metadata["counters"], resumed_metadata["counters"])

    def test_sft_skips_prompt_only_windows_and_updates(self):
        records_path = self.directory / "sft.jsonl"
        records_path.write_text(json.dumps({"prompt": "long context " * 8, "response": "Answer."}) + "\n")
        records = self.run_training(self.directory / "sft", 1, mode="sft", data=records_path)
        self.assertTrue(records[-1]["finite"])
        self.assertEqual(records[-1]["updates"], 1)
        self.assertEqual(records[-1]["samples_consumed"], 2)

    def test_token_weighted_accumulation_equals_combined_batch(self):
        tokenizer = ByteBPETokenizer()
        config = Config(vocab_size=tokenizer.vocab_size, dim=8, layers=1, heads=2,
                        q_rank=4, kv_rank=4, content_dim=2, rope_dim=2, value_dim=2, hidden_dim=12)
        samples = [next(pack_sft([record], tokenizer, 8)) for record in
                   ({"prompt": "p", "response": "a"}, {"prompt": "q", "response": "bcde"})]
        self.assertNotEqual(samples[0].supervised_tokens, samples[1].supervised_tokens)
        separate, combined = Transformer(config, 11), Transformer(config, 11)
        count = sum(sample.supervised_tokens for sample in samples)
        for sample in samples:
            loss = sft_loss(separate(sample.input_ids, sample.attention_mask), sample.targets, sample.loss_mask)
            (loss * (sample.supervised_tokens / count)).backward()
        batch = next(batch_packed(samples, 2))
        sft_loss(combined(batch.input_ids, batch.attention_mask), batch.targets, batch.loss_mask).backward()
        for name in separate.params:
            np.testing.assert_allclose(separate.params[name].grad, combined.params[name].grad,
                                       rtol=3e-5, atol=2e-6, err_msg=name)

    def test_tied_embedding_receives_lookup_and_output_gradients(self):
        config = Config(vocab_size=12, dim=8, layers=1, heads=2, q_rank=4,
                        kv_rank=4, content_dim=2, rope_dim=2, value_dim=2, hidden_dim=12)
        model = Transformer(config, 5, dtype=np.float64)
        self.assertNotIn("head", model.params)
        self.assertEqual(len({id(parameter) for parameter in model.parameters().values()}), len(model.params))
        input_ids, targets = np.array([[1, 4, 5]]), np.array([[4, 5, 2]])
        mask = np.ones_like(targets, dtype=np.float32)
        sft_loss(model(input_ids), targets, mask).backward()
        embedding = model.params["embedding"]
        index, delta = (4, 0), 1e-6
        analytical, original = float(embedding.grad[index]), embedding.data[index]
        embedding.data[index] = original + delta
        plus = float(sft_loss(model(input_ids), targets, mask).data)
        embedding.data[index] = original - delta
        minus = float(sft_loss(model(input_ids), targets, mask).data)
        embedding.data[index] = original
        self.assertAlmostEqual(analytical, (plus - minus) / (2 * delta), places=6)

    def test_forward_precision_overflow_has_specific_error(self):
        model = Transformer(Config(vocab_size=259))
        model.precision = "fp16"
        model.params["embedding"].data.fill(1e5)
        self.assertTrue(np.isfinite(model.params["embedding"].data).all())
        with self.assertRaisesRegex(FloatingPointError, "Parameter overflow.*loss scaling"):
            model(np.array([[1, 10]]))
        with self.assertRaisesRegex(FloatingPointError, "Activation overflow"):
            model.quant(Tensor(np.array([1e5], dtype=np.float32)))

    def test_forward_overflow_discards_partial_gradients_and_checkpoints_unchanged_weights(self):
        output = self.directory / "overflow-run"
        initial_parameters = {}

        class OverflowOnSecondMicrobatch(Transformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                initial_parameters.update({name: parameter.data.copy() for name, parameter in self.params.items()})
                self.calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    # Exercise the real forward precision guard after the first
                    # microbatch has already populated parameter gradients.
                    self.quant(Tensor(np.array([1e5], dtype=np.float32)))
                return super().__call__(*args, **kwargs)

        argv = [str(ROOT / "train.py"), "pretrain", "--output", str(output), "--data", str(self.corpus),
                "--steps", "2", "--sequence-length", "16", "--accumulation", "2", "--precision", "fp16",
                "--max-seconds", "5"]
        with patch.object(training_cli, "Transformer", OverflowOnSecondMicrobatch), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(FloatingPointError, "Stopped after saving unchanged weights: Activation overflow"):
                training_cli.main()
        records = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["updates"], 0)
        self.assertEqual(records[0]["skipped_updates"], 1)
        self.assertFalse(records[0]["finite"])
        self.assertIn("Activation overflow", records[0]["forward_error"])
        metadata, arrays = self.checkpoint(output)
        self.assertEqual(metadata["gradients"], [])
        self.assertEqual(metadata["optimizer"]["step_count"], 0)
        self.assertEqual(metadata["scaler"]["scale"], 64)
        for index, name in enumerate(metadata["parameters"]):
            np.testing.assert_array_equal(arrays[f"parameter_{index}"], initial_parameters[name])
            np.testing.assert_array_equal(arrays[f"master_{index}"], initial_parameters[name])
            self.assertFalse(np.any(arrays[f"m_{index}"]))
            self.assertFalse(np.any(arrays[f"v_{index}"]))


if __name__ == "__main__":
    unittest.main()
