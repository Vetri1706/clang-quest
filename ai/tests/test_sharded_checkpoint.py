"""Two-process numerical continuation, commit failure and artifact adversary tests."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distributed_demo import run_spawned
from rawllm.distributed import CollectiveServer, DistributedError, ProcessGroup
from rawllm import sharded_checkpoint as checkpoint
from rawllm.zero import ShardedAdamW


def _parameters():
    return {"matrix": np.arange(7, dtype=np.float32).reshape(1, 7) / 10,
            "tiny": np.array([0.25], dtype=np.float32)}


def _optimizer(group):
    return ShardedAdamW(_parameters(), group, learning_rate=0.01, weight_decay=0.03)


def _accumulate(optimizer, rng):
    for name, parameter in _parameters().items():
        gradient = rng.normal(size=parameter.shape).astype(np.float32) * 0.1
        optimizer.accumulate_gradient(name, gradient)


def _continue_worker(rank, host, port, token, directory):
    with ProcessGroup(rank, 2, host, port, token, timeout=5) as group:
        optimizer, rng = _optimizer(group), np.random.default_rng(100 + rank)
        _accumulate(optimizer, rng)
        optimizer.step()
        _accumulate(optimizer, rng)
        checkpoint.save_sharded_checkpoint(directory, optimizer, rng,
                                           {"rank": rank, "update": 1, "microbatch": 1}, run_id="continuation-test")
        before = optimizer.state_dict()
        _accumulate(optimizer, rng)
        optimizer.step(gradient_scale=2)
        expected = optimizer.state_dict()
        expected_random = rng.normal(size=5)
        # Loading a verified generation can repair damaged live values; those
        # values are not used as the authority for incoming state validation.
        optimizer.parameters.fill(np.inf)
        optimizer.gradients.fill(np.nan)
        optimizer.second_moment.fill(-1)
        rng = np.random.default_rng(0)
        counters = checkpoint.load_sharded_checkpoint(directory, optimizer, rng)
        restored = optimizer.state_dict()
        _accumulate(optimizer, rng)
        optimizer.step(gradient_scale=2)
        actual = optimizer.state_dict()
        actual_random = rng.normal(size=5)
        checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"rank": rank, "update": 2})
        group.barrier()
        return {"before": before, "restored": restored, "expected": expected, "actual": actual,
                "expected_random": expected_random, "actual_random": actual_random, "counters": counters}


def _save_worker(rank, host, port, token, directory, run_id, action="save"):
    with ProcessGroup(rank, 2, host, port, token, timeout=3) as group:
        optimizer, rng = _optimizer(group), np.random.default_rng(rank)
        if action == "load":
            counters = checkpoint.load_sharded_checkpoint(directory, optimizer, rng)
            group.barrier()
            return counters
        original = checkpoint._write_immutable
        if action == "kill" and rank == 1:
            def crash(path, writer):
                if path.suffix == ".json":
                    os._exit(23)
                return original(path, writer)
            checkpoint._write_immutable = crash
        try:
            checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"rank": rank}, run_id=run_id)
            result = {"saved": True}
        except checkpoint.ShardedCheckpointError as error:
            result = {"saved": False, "error": str(error)}
        finally:
            checkpoint._write_immutable = original
        group.barrier()
        return result


def _restart_worker(rank, host, port, token, directory, resume):
    with ProcessGroup(rank, 2, host, port, token, timeout=5) as group:
        optimizer, rng = _optimizer(group), np.random.default_rng(300 + rank)
        if resume:
            checkpoint.load_sharded_checkpoint(directory, optimizer, rng)
        else:
            _accumulate(optimizer, rng)
            optimizer.step()
            _accumulate(optimizer, rng)
            checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"update": 1, "microbatch": 1}, run_id="fresh-process-restart")
        _accumulate(optimizer, rng)
        optimizer.step(gradient_scale=2)
        result = {"state": optimizer.state_dict(), "random": rng.normal(size=5)}
        if resume:
            checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"update": 2, "microbatch": 0})
        group.barrier()
        return result


def _partial_write_worker(rank, host, port, token, directory):
    with ProcessGroup(rank, 2, host, port, token, timeout=5) as group:
        optimizer, rng = _optimizer(group), np.random.default_rng(rank)
        checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"saved": 1})
        previous = (Path(directory) / "manifest.json").read_bytes()
        optimizer.parameters += 3
        original = checkpoint._write_immutable
        def injected(path, writer):
            if rank == 1 and path.suffix == ".json":
                raise OSError("injected metadata write failure")
            return original(path, writer)
        checkpoint._write_immutable = injected
        failed = False
        try:
            checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"saved": 2})
        except checkpoint.ShardedCheckpointError:
            failed = True
        finally:
            checkpoint._write_immutable = original
        unchanged = (Path(directory) / "manifest.json").read_bytes() == previous
        counters = checkpoint.load_sharded_checkpoint(directory, optimizer, rng)
        group.barrier()
        return {"failed": failed, "unchanged": unchanged, "counters": counters,
                "parameters": optimizer.parameters.copy()}


def _tamper(directory, corruption):
    if corruption in ("live_shape", "live_readonly"):
        return
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text())
    shard = manifest["ranks"][1]
    payload = Path(directory) / shard["payload"]
    if corruption == "checksum":
        with payload.open("ab") as handle:
            handle.write(b"corrupted")
        return
    if corruption == "symlink":
        target = payload.with_suffix(".real")
        payload.rename(target)
        payload.symlink_to(target.name)
        return
    if corruption == "topology":
        manifest["topology"]["world_size"] = 3
    elif corruption == "path":
        shard["payload"] = "../not-a-shard.npz"
    elif corruption in ("step_bool", "step_float", "counter_tag", "rng_type", "counter_inf"):
        metadata_path = Path(directory) / shard["metadata"]
        metadata = json.loads(metadata_path.read_text())
        if corruption == "step_bool":
            metadata["optimizer"]["step"] = False
        elif corruption == "step_float":
            metadata["optimizer"]["step"] = 0.0
        elif corruption == "counter_tag":
            metadata["counters"] = {"__array__": 7, "dtype": "<i8"}
        elif corruption == "counter_inf":
            metadata["counters"] = {"overflow": "exponent-marker"}
        else:
            metadata["rng"]["bit_generator"] = "MT19937"
        raw = json.dumps(metadata).encode()
        if corruption == "counter_inf":
            raw = raw.replace(b'"exponent-marker"', b"1e999")
        metadata_path.write_bytes(raw)
        shard["metadata_sha256"] = hashlib.sha256(raw).hexdigest()
        shard["metadata_bytes"] = len(raw)
    else:
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if corruption == "moment":
            arrays["second_moment"][0] = -1
        elif corruption == "shape":
            arrays["parameters"] = np.ones(1_000_000, dtype=np.float32)
        elif corruption == "object":
            arrays["parameters"] = np.array([{"malicious": "object"}] * len(arrays["parameters"]), dtype=object)
        elif corruption == "inventory":
            arrays["unexpected"] = np.ones(1)
        else:
            raise ValueError("Unknown test corruption")
        with payload.open("wb") as handle:
            np.savez(handle, **arrays)
        raw = payload.read_bytes()
        shard["payload_sha256"] = hashlib.sha256(raw).hexdigest()
        shard["payload_bytes"] = len(raw)
    path.write_text(json.dumps(manifest))


def _corruption_worker(rank, host, port, token, directory, corruption):
    with ProcessGroup(rank, 2, host, port, token, timeout=5) as group:
        optimizer, rng = _optimizer(group), np.random.default_rng(200 + rank)
        checkpoint.save_sharded_checkpoint(directory, optimizer, rng, {"step": 0})
        if rank == 0:
            _tamper(directory, corruption)
        group.barrier()
        optimizer.parameters.fill(31 + rank)
        if corruption == "live_shape" and rank == 0:
            for name in checkpoint._ARRAYS:
                setattr(optimizer, name, np.full(1, 99, dtype=np.float32))
        if corruption == "live_readonly" and rank == 0:
            optimizer.gradients.flags.writeable = False
        before = optimizer.state_dict()
        rng_before = json.dumps(rng.bit_generator.state, sort_keys=True)
        error = None
        try:
            checkpoint.load_sharded_checkpoint(directory, optimizer, rng, max_payload_bytes=1024)
        except checkpoint.ShardedCheckpointError as caught:
            error = str(caught)
        group.barrier()
        return {"error": error, "unchanged": all(np.array_equal(getattr(optimizer, name), before[name])
                                                   for name in checkpoint._ARRAYS),
                "rng_unchanged": json.dumps(rng.bit_generator.state, sort_keys=True) == rng_before}


class ShardedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "checkpoint"

    def tearDown(self):
        self.temporary.cleanup()

    def run_workers(self, worker, *arguments):
        with CollectiveServer(2, timeout=4) as server:
            return run_spawned(worker, 2, (server.host, server.port, server.token, str(self.directory), *arguments), timeout=12)

    def test_exact_mid_accumulation_resume_and_second_generation(self):
        results = self.run_workers(_continue_worker)
        for rank, result in enumerate(results):
            for name in checkpoint._ARRAYS:
                np.testing.assert_array_equal(result["before"][name], result["restored"][name])
                np.testing.assert_array_equal(result["expected"][name], result["actual"][name])
            self.assertEqual(result["restored"]["accumulations"], result["before"]["accumulations"])
            np.testing.assert_array_equal(result["expected_random"], result["actual_random"])
            self.assertEqual(result["counters"], {"rank": rank, "update": 1, "microbatch": 1})
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(manifest["step"], 2)
        self.assertEqual(manifest["run_id"], "continuation-test")
        self.assertEqual(len(list(self.directory.glob("*.npz"))), 4)

    def test_new_process_group_restart_matches_uninterrupted_continuation(self):
        expected = self.run_workers(_restart_worker, False)
        actual = self.run_workers(_restart_worker, True)
        for first, second in zip(expected, actual):
            for name in checkpoint._ARRAYS:
                np.testing.assert_array_equal(first["state"][name], second["state"][name])
            np.testing.assert_array_equal(first["random"], second["random"])
            self.assertEqual(second["state"]["step"], 2)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(manifest["run_id"], "fresh-process-restart")
        self.assertEqual(manifest["step"], 2)

    def test_partial_rank_write_preserves_prior_commit_and_ignores_orphans(self):
        results = self.run_workers(_partial_write_worker)
        for result in results:
            self.assertTrue(result["failed"] and result["unchanged"])
            self.assertEqual(result["counters"], {"saved": 1})
        self.assertGreater(len(list(self.directory.glob("*.npz"))), 2)

    def test_corrupt_payload_is_collectively_rejected_before_mutation(self):
        for corruption in ("checksum", "moment", "shape", "object", "inventory", "symlink", "topology", "path",
                           "step_bool", "step_float", "counter_tag", "rng_type", "live_shape", "live_readonly", "counter_inf"):
            with self.subTest(corruption=corruption):
                original = self.directory
                self.directory = original.parent / corruption
                results = self.run_workers(_corruption_worker, corruption)
                self.directory = original
                for result in results:
                    self.assertIsNotNone(result["error"])
                    self.assertTrue(result["unchanged"] and result["rng_unchanged"])

    def test_unrelated_job_cannot_replace_owner_manifest(self):
        first = self.run_workers(_save_worker, "owner-one")
        self.assertTrue(all(result["saved"] for result in first))
        previous = (self.directory / "manifest.json").read_bytes()
        second = self.run_workers(_save_worker, "owner-two")
        self.assertTrue(all(not result["saved"] and "another run" in result["error"] for result in second))
        self.assertEqual((self.directory / "manifest.json").read_bytes(), previous)

    def test_live_writer_lock_rejects_concurrent_transaction(self):
        self.directory.mkdir()
        with (self.directory / ".writer.lock").open("wb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            results = self.run_workers(_save_worker, "locked-job")
            fcntl.flock(handle, fcntl.LOCK_UN)
        self.assertTrue(all(not result["saved"] for result in results))
        self.assertFalse((self.directory / "manifest.json").exists())

    def test_killed_rank_before_ack_preserves_previous_generation(self):
        self.run_workers(_save_worker, "kill-test")
        previous = (self.directory / "manifest.json").read_bytes()
        with self.assertRaises(DistributedError):
            self.run_workers(_save_worker, "kill-test", "kill")
        self.assertEqual((self.directory / "manifest.json").read_bytes(), previous)
        results = self.run_workers(_save_worker, "kill-test", "load")
        self.assertEqual(results, [{"rank": 0}, {"rank": 1}])


if __name__ == "__main__":
    unittest.main()
