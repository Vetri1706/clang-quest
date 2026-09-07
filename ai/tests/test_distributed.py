"""Independent numerical references plus transport and worker-failure tests."""
from __future__ import annotations

import hashlib
import hmac
import json
import multiprocessing as mp
import socket
import struct
import time
import unittest

import numpy as np

from distributed_demo import run_demo, run_spawned
from rawllm.distributed import (CollectiveServer, DistributedError, ParameterServerClient,
                                ParallelTopology, ProcessGroup, ProtocolError,
                                _recv_frame, _send_frame)
from rawllm.zero import ShardedAdamW, shard_bounds


def _collectives_worker(rank, host, port, token):
    with ProcessGroup(rank, 3, host, port, token) as group:
        summed = group.all_reduce(np.full(5, rank + 1, dtype=np.float64))
        mean = group.all_reduce(np.full(2, rank + 1, dtype=np.float32), "mean")
        gathered = group.all_gather(np.arange(rank + 1, dtype=np.int32))
        scattered = group.scatter([np.full(r + 1, r + 4, dtype=np.int64) for r in range(3)] if rank == 1 else None, root=1)
        root_gathered = group.gather(np.array([rank], dtype=np.float64), root=2)
        reduced = group.reduce_scatter(np.arange(7, dtype=np.float32) + rank)
        broadcast = group.broadcast(np.array(17, dtype=np.int64) if rank == 2 else None, root=2)
        subgroup = None
        if rank in (0, 2):
            subgroup = group.subgroup([0, 2]).all_reduce(np.array([rank + 1], dtype=np.float32))
        if rank == 0:
            group.send(np.array([31, 41], dtype=np.int32), 2, "explicit-message")
        point = group.recv(0, "explicit-message") if rank == 2 else None
        group.barrier()
        return {"sum": summed, "mean": mean, "all_gather": gathered, "scatter": scattered,
                "gather": root_gathered, "reduce_scatter": reduced, "broadcast": broadcast,
                "subgroup": subgroup, "point": point}


def _parameter_worker(rank, host, port, token):
    with ProcessGroup(rank, 3, host, port, token) as group:
        client = ParameterServerClient(group)
        if rank == 0:
            client.initialize("weights", np.array([1, 2, 3], dtype=np.float32))
        group.barrier()
        version, initial = client.pull("weights")
        accepted, new_version, updated = client.push("weights", version, np.full(3, rank + 1, dtype=np.float32), 0.1)
        stale = client.push("weights", version, np.zeros(3, dtype=np.float32), 0.1)
        group.barrier()
        return initial, accepted, new_version, updated, stale


def _initial_parameters():
    return {"matrix": np.arange(14, dtype=np.float32).reshape(2, 7) / 20,
            "bias": np.array([0.2, -0.3], dtype=np.float32)}


def _zero_worker(rank, host, port, token):
    with ProcessGroup(rank, 3, host, port, token) as group:
        initial = _initial_parameters()
        optimizer = ShardedAdamW(initial, group, learning_rate=0.01, weight_decay=0.1)
        del initial
        materialized = {}
        statistics = []
        for step in range(3):
            for name, parameter in _initial_parameters().items():
                full = optimizer.materialize(name)
                materialized[name] = full.copy()
                # Two microbatch contributions with distinct rank-dependent data.
                for micro in range(2):
                    gradient = np.full(parameter.shape, (rank + 1) * (step + 1) * (micro + 1) * 0.01, dtype=np.float32)
                    gradient += np.arange(parameter.size, dtype=np.float32).reshape(parameter.shape) * 0.001
                    optimizer.accumulate_gradient(name, gradient)
                optimizer.release(name)
                del full, gradient
            statistics.append(optimizer.step(max_grad_norm=0.3, gradient_scale=2.0))
        for name in optimizer.partitions:
            full = optimizer.materialize(name)
            materialized[name] = full.copy()
            optimizer.release(name)
            del full
        snapshot = optimizer.state_dict()
        optimizer.parameters.fill(100)
        optimizer.load_state_dict(snapshot)
        restored = optimizer.parameters.copy()
        group.barrier()
        return {"parameters": materialized, "local": restored, "state": snapshot,
                "bytes": optimizer.persistent_bytes, "statistics": statistics}


def _nonfinite_zero_worker(rank, host, port, token):
    with ProcessGroup(rank, 2, host, port, token) as group:
        optimizer = ShardedAdamW({"weight": np.ones(3, dtype=np.float32)}, group)
        gradient = np.ones(3, dtype=np.float32)
        if rank == 1:
            gradient[0] = np.inf
        optimizer.accumulate_gradient("weight", gradient)
        report = optimizer.step()
        group.barrier()
        return report, optimizer.parameters.copy(), optimizer.step_number


def _mismatched_shape_worker(rank, host, port, token):
    with ProcessGroup(rank, 2, host, port, token) as group:
        return group.all_reduce(np.ones(rank + 1, dtype=np.float32))


def _failing_worker(rank):
    if rank == 0:
        raise RuntimeError("intentional worker failure")
    time.sleep(10)
    return rank


class DistributedTests(unittest.TestCase):
    def test_collectives_subgroups_and_point_to_point(self):
        with CollectiveServer(3) as server:
            results = run_spawned(_collectives_worker, 3, (server.host, server.port, server.token))
        for rank, result in enumerate(results):
            np.testing.assert_array_equal(result["sum"], np.full(5, 6))
            np.testing.assert_array_equal(result["mean"], np.full(2, 2))
            for other, array in enumerate(result["all_gather"]):
                np.testing.assert_array_equal(array, np.arange(other + 1))
            np.testing.assert_array_equal(result["scatter"], np.full(rank + 1, rank + 4))
            expected_shard = np.array_split(np.arange(7) * 3 + 3, 3)[rank]
            np.testing.assert_array_equal(result["reduce_scatter"], expected_shard)
            self.assertEqual(result["broadcast"].shape, ())
            self.assertEqual(result["broadcast"].item(), 17)
            if rank in (0, 2):
                np.testing.assert_array_equal(result["subgroup"], [4])
            if rank != 2:
                self.assertIsNone(result["gather"])
        for rank, value in enumerate(results[2]["gather"]):
            np.testing.assert_array_equal(value, [rank])
        np.testing.assert_array_equal(results[2]["point"], [31, 41])

    def test_parameter_server_atomic_mean_and_stale_rejection(self):
        with CollectiveServer(3) as server:
            results = run_spawned(_parameter_worker, 3, (server.host, server.port, server.token))
        for initial, accepted, version, updated, stale in results:
            np.testing.assert_array_equal(initial, [1, 2, 3])
            self.assertTrue(accepted)
            self.assertEqual(version, 1)
            np.testing.assert_allclose(updated, [0.8, 1.8, 2.8])
            self.assertEqual(stale, (False, 1, None))

    def test_zero_stage3_matches_unsharded_adamw_with_uneven_shards(self):
        with CollectiveServer(3) as server:
            results = run_spawned(_zero_worker, 3, (server.host, server.port, server.token))
        expected = _initial_parameters()
        moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in expected.items()}
        expected_norms = []
        for step in range(1, 4):
            gradients = {name: np.full(value.shape, 0.03 * step, dtype=np.float32)
                         + np.arange(value.size, dtype=np.float32).reshape(value.shape) * 0.001
                         for name, value in expected.items()}
            norm = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in gradients.values()))
            expected_norms.append(norm)
            clip = min(1.0, 0.3 / (norm + 1e-12))
            for name in expected:
                gradient = gradients[name] * np.float32(clip)
                m, v = moments[name]
                m[:] = 0.9 * m + 0.1 * gradient
                v[:] = 0.999 * v + 0.001 * gradient * gradient
                expected[name] = expected[name] * (1 - 0.01 * 0.1)
                expected[name] -= 0.01 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        total_local = 0
        for rank, result in enumerate(results):
            local_pieces = []
            for name, value in expected.items():
                np.testing.assert_allclose(result["parameters"][name], value, atol=3e-7, rtol=2e-6)
                start, stop = shard_bounds(value.size, rank, 3)
                local_pieces.append(value.reshape(-1)[start:stop])
            np.testing.assert_allclose(result["local"], np.concatenate(local_pieces), atol=3e-7)
            self.assertEqual(result["bytes"], result["local"].size * 16)
            total_local += result["local"].size
            for report, norm in zip(result["statistics"], expected_norms):
                self.assertTrue(report["updated"])
                self.assertAlmostEqual(report["grad_norm"], norm, places=7)
        self.assertEqual(total_local, 16)
        self.assertEqual(results[2]["state"]["parameters"].size, 4)

    def test_zero_nonfinite_gradient_skips_all_ranks_atomically(self):
        with CollectiveServer(2) as server:
            results = run_spawned(_nonfinite_zero_worker, 2, (server.host, server.port, server.token))
        for report, shard, step in results:
            self.assertFalse(report["updated"])
            np.testing.assert_array_equal(shard, np.ones_like(shard))
            self.assertEqual(step, 0)

    def test_eight_process_3d_matches_single_process_reference(self):
        report = run_demo()
        self.assertTrue(report["verified"])
        self.assertLess(report["maximum_gradient_absolute_error"], 1e-11)

    def test_real_mla_transformer_dp_matches_combined_batch(self):
        from distributed_train import run_training
        report = run_training(world_size=2, steps=2)
        self.assertTrue(report["verified"])
        self.assertEqual(report["maximum_replica_absolute_error"], 0)
        self.assertLess(report["maximum_weight_absolute_error"], 2e-6)

    def test_real_mla_transformer_3d_matches_all_parameters(self):
        from distributed_llm3d import run_training
        report = run_training(steps=2)
        self.assertTrue(report["verified"])
        self.assertEqual(report["topology"], {"data": 2, "pipeline": 2, "tensor": 2, "processes": 8})
        self.assertTrue(report["tensor_replicas_exact"])
        self.assertEqual(report["maximum_data_replica_absolute_error"], 0)
        self.assertLess(report["maximum_gradient_absolute_error"], 2e-6)
        self.assertLess(report["maximum_weight_absolute_error"], 2e-6)
        self.assertTrue(all(owned < report["parameters"] for owned in report["locally_owned_parameters_by_rank"]))

    def test_mismatched_collective_shapes_fail_without_hanging(self):
        with CollectiveServer(2, timeout=2) as server:
            with self.assertRaisesRegex(DistributedError, "matching numeric shapes"):
                run_spawned(_mismatched_shape_worker, 2, (server.host, server.port, server.token), timeout=5)

    def test_missing_rank_deadline(self):
        start = time.monotonic()
        with CollectiveServer(2, timeout=0.3) as server:
            with ProcessGroup(0, 2, server.host, server.port, server.token, timeout=1) as group:
                with self.assertRaisesRegex(DistributedError, "timed out"):
                    group.all_reduce(np.ones(1, dtype=np.float32))
        self.assertLess(time.monotonic() - start, 2)

    def test_bad_authentication_is_rejected(self):
        with CollectiveServer(1, timeout=1) as server:
            with self.assertRaises(ProtocolError):
                ProcessGroup(0, 1, server.host, server.port, "invalid-token-" * 4, timeout=1)

    def test_worker_error_terminates_other_processes(self):
        start = time.monotonic()
        with self.assertRaisesRegex(DistributedError, "intentional worker failure"):
            run_spawned(_failing_worker, 2, (), timeout=4)
        self.assertLess(time.monotonic() - start, 4)
        self.assertFalse([p for p in mp.active_children() if p.name.startswith("rawllm-rank-")])

    def test_topology_groups(self):
        topology = ParallelTopology(2, 2, 2)
        self.assertEqual(topology.groups(5), {"data": (1, 5), "pipeline": (5, 7), "tensor": (4, 5)})
        for rank in range(8):
            self.assertEqual(topology.rank(*topology.coordinates(rank)), rank)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.first, self.second = socket.socketpair()
        self.first.settimeout(0.5)
        self.second.settimeout(0.5)
        self.key = b"test-shared-key-at-least-32-bytes-long"

    def tearDown(self):
        self.first.close()
        self.second.close()

    def test_wire_roundtrip_big_endian_noncontiguous_and_empty(self):
        array = np.arange(12, dtype=">f4").reshape(3, 4)[:, ::2]
        _send_frame(self.first, self.key, 0, {"op": "test"}, [array, np.empty(0, dtype=np.float32)], 1024)
        message, arrays = _recv_frame(self.second, self.key, 0, 1024)
        self.assertEqual(message, {"op": "test"})
        np.testing.assert_array_equal(arrays[0], array)
        self.assertEqual(arrays[1].shape, (0,))

    def test_replay_sequence_rejected(self):
        _send_frame(self.first, self.key, 3, {}, [], 1024)
        with self.assertRaisesRegex(ProtocolError, "sequence"):
            _recv_frame(self.second, self.key, 4, 1024)

    def test_outgoing_size_and_object_dtype_rejected(self):
        with self.assertRaisesRegex(ProtocolError, "bound"):
            _send_frame(self.first, self.key, 0, {}, [np.ones(10, dtype=np.float64)], 16)
        with self.assertRaisesRegex(ProtocolError, "dtype"):
            _send_frame(self.first, self.key, 0, {}, [np.array([object()], dtype=object)], 1024)

    def test_incoming_oversize_rejected_before_body_read(self):
        self.first.sendall(struct.pack("!IQ", 10, 1000000))
        with self.assertRaisesRegex(ProtocolError, "bound"):
            _recv_frame(self.second, self.key, 0, 1024)

    def test_authenticated_forged_dtype_is_rejected(self):
        metadata = json.dumps({"seq": 0, "message": {}, "arrays": [{"dtype": "|O", "shape": [0], "offset": 0, "nbytes": 0}]}).encode()
        header = struct.pack("!IQ", len(metadata), 0)
        digest = hmac.new(self.key, header + metadata, hashlib.sha256).digest()
        self.first.sendall(header + metadata + digest)
        with self.assertRaisesRegex(ProtocolError, "dtype"):
            _recv_frame(self.second, self.key, 0, 1024)

    def test_tampered_payload_is_rejected(self):
        metadata = json.dumps({"seq": 0, "message": {}, "arrays": []}).encode()
        header = struct.pack("!IQ", len(metadata), 0)
        self.first.sendall(header + metadata + bytes(32))
        with self.assertRaisesRegex(ProtocolError, "authentication"):
            _recv_frame(self.second, self.key, 0, 1024)


if __name__ == "__main__":
    unittest.main()
