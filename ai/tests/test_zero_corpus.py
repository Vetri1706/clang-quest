"""Corpus/SFT target weighting, exact process restart, and metadata rejection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from rawllm.data import batch_packed
from rawllm.model import Config, Transformer
from rawllm.optim import AdamW, clip_grad_norm
from rawllm.tensor import cross_entropy
from rawllm.tokenizer import ByteBPETokenizer
from rawllm.zero import shard_bounds
from zero_corpus import MAX_SOURCE_BYTES, PackedReplay, packed_samples, run_training


def checkpoint_arrays(directory: Path):
    manifest = json.loads((directory / "checkpoint" / "manifest.json").read_text())
    shards = []
    for descriptor in manifest["ranks"]:
        # Inputs here are test-generated, bounded local archives. Production
        # checkpoint loading uses safeio's header-before-allocation validation.
        with np.load(directory / "checkpoint" / descriptor["payload"], allow_pickle=False) as archive:
            shards.append({name: archive[name].copy() for name in archive.files})
    return manifest, shards


class ZeroCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "corpus.txt"
        self.source.write_text("A C++ object owns its resources through RAII.\n"
                               "A vector stores elements contiguously and checks its size.\n"
                               "Compile source to an intermediate representation before machine code.\n", encoding="utf8")

    def test_uninterrupted_and_new_process_resume_have_exact_shards(self):
        complete, resumed = self.root / "complete", self.root / "resumed"
        whole = run_training(complete, source=self.source, steps=2)
        initial = run_training(resumed, source=self.source, steps=1)
        continued = run_training(resumed, source=self.source, steps=2, resume=True)
        self.assertEqual(initial["completed_steps"], 1)
        self.assertEqual(continued["starting_step"], 1)
        self.assertEqual(continued["completed_steps"], 2)
        self.assertEqual(whole["global_target_tokens"], continued["global_target_tokens"])
        _, expected = checkpoint_arrays(complete)
        _, actual = checkpoint_arrays(resumed)
        for expected_rank, actual_rank in zip(expected, actual):
            for name in expected_rank:
                np.testing.assert_array_equal(actual_rank[name], expected_rank[name])

    def test_sft_unequal_target_mass_matches_dense_global_token_objective(self):
        source = self.root / "sft.jsonl"
        records = [{"prompt": "Q:", "response": "A"},
                   {"prompt": "Another question with a longer prompt:",
                    "response": "A compiler turns source into machine instructions. RAII manages resources."}]
        source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf8")
        directory = self.root / "sft"
        report = run_training(directory, mode="sft", source=source, steps=1,
                              accumulation=1, sequence_length=12)
        masses = [rank[0]["local"] for rank in report["token_mass_by_rank"]]
        self.assertNotEqual(masses[0], masses[1])
        settings = json.loads((directory / "settings.json").read_text())
        tokenizer = ByteBPETokenizer.load(directory / "tokenizer.json")
        stream = PackedReplay(lambda: packed_samples(settings, directory, tokenizer))
        batches = [stream.next(), stream.next()]
        combined = next(batch_packed(batches, 2))
        model = Transformer(Config(**settings["config"]), seed=settings["seed"])
        optimizer = AdamW(model.parameters(), lr=settings["learning_rate"])
        loss = cross_entropy(model(combined.input_ids, combined.attention_mask), combined.targets,
                             mask=combined.loss_mask)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        self.assertAlmostEqual(report["losses"][0], loss.item(), places=5)
        _, actual = checkpoint_arrays(directory)
        for rank in range(2):
            pieces = []
            for parameter in model.parameters().values():
                start, stop = shard_bounds(parameter.size, rank, 2)
                pieces.append(parameter.data.reshape(-1)[start:stop])
            np.testing.assert_allclose(actual[rank]["parameters"], np.concatenate(pieces), atol=2e-7, rtol=2e-5)

    def test_changed_original_or_staged_data_is_rejected_before_workers(self):
        directory = self.root / "data_validation"
        run_training(directory, source=self.source, steps=1)
        self.source.write_text("modified original", encoding="utf8")
        with patch("zero_corpus.run_spawned") as workers:
            with self.assertRaisesRegex(ValueError, "dataset digest"):
                run_training(directory, source=self.source, steps=2, resume=True)
            workers.assert_not_called()
        (directory / "source.txt").write_text("modified snapshot", encoding="utf8")
        with patch("zero_corpus.run_spawned") as workers:
            with self.assertRaisesRegex(ValueError, "staged dataset digest"):
                run_training(directory, steps=2, resume=True)
            workers.assert_not_called()

    def test_counter_tampering_with_valid_checksums_is_rejected_before_workers(self):
        directory = self.root / "counter_validation"
        run_training(directory, source=self.source, steps=1)
        manifest_path = directory / "checkpoint" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        descriptor = manifest["ranks"][0]
        metadata_path = manifest_path.parent / descriptor["metadata"]
        metadata = json.loads(metadata_path.read_text())
        metadata["counters"]["samples_consumed"] += 1
        raw = json.dumps(metadata).encode()
        metadata_path.write_bytes(raw)
        descriptor["metadata_sha256"] = hashlib.sha256(raw).hexdigest()
        descriptor["metadata_bytes"] = len(raw)
        manifest_path.write_text(json.dumps(manifest))
        original_manifest = manifest_path.read_bytes()
        with patch("zero_corpus.run_spawned") as workers:
            with self.assertRaisesRegex(ValueError, "counters, replay position"):
                run_training(directory, source=self.source, steps=2, resume=True)
            workers.assert_not_called()
        self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_tokenizer_settings_and_source_bounds_are_checked(self):
        directory = self.root / "settings_validation"
        run_training(directory, source=self.source, steps=1)
        with self.assertRaisesRegex(ValueError, "settings changed"):
            run_training(directory, source=self.source, steps=2, sequence_length=8, resume=True)
        tokenizer_path = directory / "tokenizer.json"
        tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "tokenizer digest"):
            run_training(directory, source=self.source, steps=2, resume=True)
        oversized = self.root / "oversized.txt"
        with oversized.open("wb") as handle:
            handle.truncate(MAX_SOURCE_BYTES + 1)
        with self.assertRaises((ValueError, MemoryError)):
            run_training(self.root / "too_large", source=oversized, steps=1)

    def test_missing_resume_manifest_cannot_silently_restart_training(self):
        directory = self.root / "lost_manifest"
        run_training(directory, source=self.source, steps=1)
        manifest = directory / "checkpoint" / "manifest.json"
        manifest.rename(manifest.with_name("lost-manifest.json"))
        with patch("zero_corpus.run_spawned") as workers:
            with self.assertRaisesRegex(ValueError, "requires a committed checkpoint manifest"):
                run_training(directory, source=self.source, steps=2, resume=True)
            workers.assert_not_called()
        self.assertFalse(manifest.exists())

    def test_cli_sigterm_exits_promptly_and_checkpoint_resumes(self):
        directory = self.root / "terminated"
        project = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(project / "zero_corpus.py"), "pretrain", "--output", str(directory),
                   "--data", str(self.source), "--steps", "100", "--sequence-length", "64", "--accumulation", "4"]
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        manifest = directory / "checkpoint" / "manifest.json"
        try:
            deadline = time.monotonic() + 10
            while not manifest.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(manifest.exists(), "CLI did not publish its first checkpoint")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=8)
            self.assertNotEqual(process.returncode, 0)
            committed = json.loads(manifest.read_text())["step"]
            self.assertLess(committed, 100)
            continued = run_training(directory, source=self.source, steps=committed + 1, resume=True,
                                     sequence_length=64, accumulation=4)
            self.assertEqual(continued["starting_step"], committed)
            self.assertEqual(continued["completed_steps"], committed + 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
