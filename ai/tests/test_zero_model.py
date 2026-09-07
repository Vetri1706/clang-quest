"""Numerical and actual full-weight lifetime checks for automatic ZeRO-3."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import weakref

import numpy as np

from distributed_train import configuration, training_batch
from rawllm.distributed import CollectiveServer, ProcessGroup
from rawllm.model import Transformer
from rawllm.tensor import Tensor, cross_entropy
from rawllm.zero_model import ZeroTransformer, initialized_parameters
from zero_train import run_training


class ZeroTransformerTests(unittest.TestCase):
    def single_rank_model(self):
        resources = ExitStack()
        self.addCleanup(resources.close)
        server = resources.enter_context(CollectiveServer(1))
        group = resources.enter_context(ProcessGroup(0, 1, server.host, server.port, server.token))
        return ZeroTransformer(configuration(), group, seed=321)

    def test_streamed_initialization_matches_regular_model_exactly(self):
        config = configuration()
        expected = Transformer(config, seed=321)
        names = []
        for name, value in initialized_parameters(config, seed=321):
            names.append(name)
            np.testing.assert_array_equal(value, expected.parameters()[name].data)
        self.assertEqual(names, list(config.shapes()))

    def test_two_rank_tied_training_matches_dense_adamw_over_three_updates(self):
        report = run_training(world_size=2, steps=3, tied=True)
        self.assertTrue(report["verified"])
        self.assertLess(report["maximum_absolute_errors"]["weights"], 3e-5)
        self.assertEqual(sum(item["locally_owned_parameters"] for item in report["memory_by_rank"]), report["parameters"])
        for memory in report["memory_by_rank"]:
            self.assertEqual(memory["maximum_simultaneous_materialized_tensors"], 1)
            self.assertIsNone(memory["active_materialization"])

    def test_three_rank_untied_training_with_uneven_shards(self):
        report = run_training(world_size=3, steps=2, tied=False)
        self.assertTrue(report["verified"])
        counts = [item["locally_owned_parameters"] for item in report["memory_by_rank"]]
        self.assertEqual(sum(counts), report["parameters"])
        self.assertGreater(max(counts), min(counts))

    def test_sharded_checkpoint_resume_matches_uninterrupted_dense_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "shards")
            first = run_training(world_size=2, steps=1, checkpoint=checkpoint)
            continued = run_training(world_size=2, steps=2, checkpoint=checkpoint, resume=checkpoint)
            self.assertTrue(first["verified"])
            self.assertTrue(continued["verified"])
            self.assertEqual(continued["starting_step"], 1)
            self.assertEqual(continued["completed_steps"], 3)
            self.assertLess(continued["maximum_absolute_errors"]["weights"], 3e-5)

    def test_autograd_graph_retains_no_full_materialized_weights(self):
        model = self.single_rank_model()
        materialized_references = []
        original = model.optimizer.materialize

        def audited_materialize(name):
            value = original(name)
            materialized_references.append(weakref.ref(value))
            return value

        model.optimizer.materialize = audited_materialize
        inputs, targets = training_batch(0, 1)
        logits = model(inputs)
        forward_calls = len(materialized_references)
        self.assertGreater(forward_calls, 10)
        self.assertTrue(all(reference() is None for reference in materialized_references))
        loss = cross_entropy(logits, targets)
        loss.backward()
        self.assertGreater(len(materialized_references), forward_calls)
        self.assertTrue(all(reference() is None for reference in materialized_references))
        self.assertIsNone(model.optimizer._active_name)
        self.assertEqual(model._anchor.grad.shape, ())
        self.assertEqual(model.optimizer._accumulations["embedding"], 2)

    def test_backward_exception_releases_current_parameter(self):
        model = self.single_rank_model()
        inputs, targets = training_batch(0, 1)
        loss = cross_entropy(model(inputs), targets)

        def rejected_gradient(name, gradient):
            raise ArithmeticError("injected gradient failure")

        model._record_gradient = rejected_gradient
        with self.assertRaisesRegex(ArithmeticError, "injected gradient failure"):
            loss.backward()
        self.assertIsNone(model.optimizer._active_name)
        self.assertIsNone(model.optimizer._active_value)

    def test_checkpoint_restore_invalidates_existing_autograd_graph(self):
        model = self.single_rank_model()
        inputs, targets = training_batch(0, 1)
        loss = cross_entropy(model(inputs), targets)
        snapshot = model.optimizer.state_dict()
        model.optimizer.load_state_dict(snapshot)
        with self.assertRaisesRegex(RuntimeError, "after its parameter shards were updated"):
            loss.backward()
        self.assertIsNone(model.optimizer._active_name)

    def test_configuration_memory_bound_fails_before_array_allocation(self):
        model = self.single_rank_model()
        with self.assertRaisesRegex(MemoryError, "largest materialized"):
            ZeroTransformer(configuration(), model.group, max_parameter_bytes=4)
        with self.assertRaisesRegex(MemoryError, "persistent"):
            ZeroTransformer(configuration(), model.group, max_persistent_bytes=4)
        huge = replace(configuration(), dim=2**63, vocab_size=2**63)
        with self.assertRaisesRegex(MemoryError, "largest materialized"):
            ZeroTransformer(huge, model.group)

    def test_batched_linear_and_transposed_tied_projection_gradients(self):
        model = self.single_rank_model()
        regular = Transformer(configuration(), seed=321)
        rng = np.random.default_rng(7)
        for name, transpose in (("blocks.0.q_down", False), ("embedding", True)):
            with self.subTest(parameter=name):
                model.zero_grad()
                parameter = regular.parameters()[name]
                parameter.zero_grad()
                inputs = rng.normal(size=(3, 5, 16)).astype(np.float32)
                x, reference_x = Tensor(inputs, requires_grad=True), Tensor(inputs.copy(), requires_grad=True)
                output = model.linear(x, name, transpose_weight=transpose)
                expected = reference_x @ (parameter.T if transpose else parameter)
                upstream = rng.normal(size=output.shape).astype(np.float32)
                output.backward(upstream)
                expected.backward(upstream)
                np.testing.assert_allclose(x.grad, reference_x.grad, atol=1e-6, rtol=1e-5)
                partition = model.optimizer.partitions[name]
                actual = model.optimizer.gradients[partition.local_start:partition.local_stop].reshape(partition.shape)
                np.testing.assert_allclose(actual, parameter.grad, atol=3e-6, rtol=2e-5)

    def test_masked_forward_and_backward_match_regular_model(self):
        model = self.single_rank_model()
        regular = Transformer(configuration(), seed=321)
        inputs, targets = training_batch(0, 1)
        segments = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        mask = (segments[:, None] == segments[None, :])[None, None]
        zero_logits, regular_logits = model(inputs, mask), regular(inputs, mask)
        np.testing.assert_allclose(zero_logits.data, regular_logits.data, atol=2e-7, rtol=2e-6)
        cross_entropy(zero_logits, targets).backward()
        cross_entropy(regular_logits, targets).backward()
        expected = np.concatenate([parameter.grad.reshape(-1) for parameter in regular.parameters().values()])
        np.testing.assert_allclose(model.optimizer.gradients, expected, atol=2e-7, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
