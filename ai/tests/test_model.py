"""Independent architecture, analytical-gradient, and paged-inference checks."""

import unittest
from unittest import mock
import numpy as np

from rawllm.model import Config, Transformer
from rawllm.cache import PagedLatentCache
from rawllm.tensor import cross_entropy, no_grad


def tiny_config(**overrides):
    fields = dict(vocab_size=17, dim=8, layers=2, heads=2, q_rank=4,
                  kv_rank=3, content_dim=2, rope_dim=2, value_dim=2,
                  hidden_dim=12, max_seq_len=24)
    fields.update(overrides)
    return Config(**fields)


def logical_cache(cache, sequence_id):
    """Copy only tokens visible to the sequence, independent of physical pages."""
    result = []
    for layer in range(cache.config.layers):
        pages = list(cache.pages(sequence_id, layer))
        result.append((np.concatenate([x for x, _ in pages]).copy(),
                       np.concatenate([x for _, x in pages]).copy()))
    return result


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.model = Transformer(self.config, seed=19, dtype=np.float64)

    def assert_cache_invariants(self, cache):
        referenced = np.zeros(cache.max_pages, dtype=np.int64)
        for sequence_id, pages in cache.tables.items():
            expected_pages = (cache.length(sequence_id) + cache.page_size - 1) // cache.page_size
            self.assertEqual(len(pages), expected_pages)
            for page in pages:
                referenced[page] += 1
        np.testing.assert_equal(cache.refs, referenced)
        self.assertEqual(len(cache.free), len(set(cache.free)))
        self.assertEqual(set(cache.free), set(np.flatnonzero(cache.refs == 0)))

    def test_parameter_count_matches_actual_tied_and_untied_models(self):
        for tied in (False, True):
            config = tiny_config(tie_embeddings=tied)
            model = Transformer(config, dtype=np.float64)
            self.assertEqual(config.parameter_count, sum(p.data.size for p in model.parameters().values()))
            self.assertEqual(config.shapes(), {name: value.shape for name, value in model.parameters().items()})
            expected = config.vocab_size * config.dim
            if not tied:
                expected *= 2
            block = (2 * config.dim + config.dim * config.q_rank + config.q_rank
                     + config.q_rank * config.heads * (config.content_dim + config.rope_dim)
                     + config.dim * config.kv_rank + config.kv_rank
                     + config.kv_rank * config.heads * (config.content_dim + config.value_dim)
                     + config.dim * config.rope_dim + config.heads * config.value_dim * config.dim
                     + 3 * config.dim * config.hidden_dim)
            self.assertEqual(config.parameter_count, expected + config.layers * block + config.dim)

    def test_seven_billion_metadata_and_guard_before_allocation(self):
        config = Config.seven_b()
        self.assertEqual(config.parameter_count, 7_018_450_944)
        with mock.patch("rawllm.model.np.random.default_rng", side_effect=AssertionError("allocation began")):
            with self.assertRaises(MemoryError):
                Transformer(config)

    def test_selected_mla_and_swiglu_parameter_gradients(self):
        ids = np.array([[1, 4, 3, 6]])
        targets = np.array([[4, 3, 6, 2]])
        loss = cross_entropy(self.model(ids), targets)
        loss.backward()
        parameter_names = ["blocks.0.q_down", "blocks.0.k_rope", "blocks.0.kv_down",
                           "blocks.0.gate", "blocks.1.q_up", "embedding"]
        epsilon = 1e-6
        for name in parameter_names:
            parameter = self.model.parameters()[name]
            self.assertTrue(np.isfinite(parameter.grad).all())
            locations = [(0, 0), (parameter.shape[0] - 1, parameter.shape[1] - 1)]
            for index in locations:
                with self.subTest(parameter=name, index=index):
                    expected = parameter.grad[index]
                    original = parameter.data[index]
                    with no_grad():
                        parameter.data[index] = original + epsilon
                        plus = cross_entropy(self.model(ids), targets).item()
                        parameter.data[index] = original - epsilon
                        minus = cross_entropy(self.model(ids), targets).item()
                    parameter.data[index] = original
                    numerical = (plus - minus) / (2 * epsilon)
                    np.testing.assert_allclose(expected, numerical, atol=3e-8, rtol=2e-4)

    def test_paged_decode_matches_full_forward_across_page_boundaries(self):
        tokens = [1, 3, 5, 2, 6, 8, 4]
        for tied in (False, True):
            model = Transformer(tiny_config(tie_embeddings=tied), seed=23, dtype=np.float64)
            cache = PagedLatentCache(model.config, page_size=2, max_pages=5, dtype=np.float64)
            for index, token in enumerate(tokens):
                with self.subTest(tied=tied, token_index=index):
                    decoded = model.decode(token, cache)
                    with no_grad():
                        expected = model(np.array([tokens[:index + 1]])).data[0, -1]
                    np.testing.assert_allclose(decoded, expected, atol=1e-12, rtol=1e-12)
                    self.assert_cache_invariants(cache)
            self.assertEqual(len(cache.tables["default"]), 4)

    def test_fork_copy_on_write_preserves_both_branches(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=5, dtype=np.float64)
        prefix = [1, 2, 3]
        for token in prefix:
            self.model.decode(token, cache, "source")
        before = logical_cache(cache, "source")
        source_pages = list(cache.tables["source"])
        cache.fork("source", "branch")
        np.testing.assert_equal(cache.refs[source_pages], [2, 2])
        branch_output = self.model.decode(4, cache, "branch")
        self.assertEqual(cache.length("source"), len(prefix))
        self.assertEqual(cache.tables["source"], source_pages)
        self.assertNotEqual(cache.tables["source"][-1], cache.tables["branch"][-1])
        after = logical_cache(cache, "source")
        for old_layer, new_layer in zip(before, after):
            for old_values, new_values in zip(old_layer, new_layer):
                np.testing.assert_equal(old_values, new_values)
        branch_before = logical_cache(cache, "branch")
        source_output = self.model.decode(5, cache, "source")
        for old_layer, new_layer in zip(branch_before, logical_cache(cache, "branch")):
            for old_values, new_values in zip(old_layer, new_layer):
                np.testing.assert_equal(old_values, new_values)
        with no_grad():
            np.testing.assert_allclose(branch_output, self.model(np.array([prefix + [4]])).data[0, -1], atol=1e-12)
            np.testing.assert_allclose(source_output, self.model(np.array([prefix + [5]])).data[0, -1], atol=1e-12)
        self.assert_cache_invariants(cache)
        cache.release("branch")
        self.assert_cache_invariants(cache)
        cache.release("source")
        self.assertEqual(len(cache.free), cache.max_pages)
        np.testing.assert_equal(cache.latent, 0)
        np.testing.assert_equal(cache.rotary, 0)

    def test_cache_exhaustion_release_and_reuse(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=1, dtype=np.float64)
        self.model.decode(1, cache)
        self.model.decode(2, cache)
        before = logical_cache(cache, "default")
        with self.assertRaises(MemoryError):
            self.model.decode(3, cache)
        self.assertEqual(cache.length("default"), 2)
        for old_layer, new_layer in zip(before, logical_cache(cache, "default")):
            for old_values, new_values in zip(old_layer, new_layer):
                np.testing.assert_equal(old_values, new_values)
        self.assert_cache_invariants(cache)
        cache.release("default")
        output = self.model.decode(4, cache, "fresh")
        with no_grad():
            np.testing.assert_allclose(output, self.model(np.array([[4]])).data[0, 0], atol=1e-12)
        self.assert_cache_invariants(cache)

    def test_failed_decode_rolls_back_page_boundary_allocation(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=3, dtype=np.float64)
        for token in [1, 2]:
            self.model.decode(token, cache)
        before = logical_cache(cache, "default")
        free_before = len(cache.free)
        with mock.patch("rawllm.cache.paged_latent_attention", side_effect=RuntimeError("injected inference error")):
            with self.assertRaisesRegex(RuntimeError, "injected inference error"):
                self.model.decode(3, cache)
        self.assertEqual(cache.length("default"), 2)
        self.assertEqual(len(cache.free), free_before)
        for old_layer, new_layer in zip(before, logical_cache(cache, "default")):
            for old_values, new_values in zip(old_layer, new_layer):
                np.testing.assert_equal(old_values, new_values)
        self.assert_cache_invariants(cache)
        output = self.model.decode(3, cache)
        with no_grad():
            np.testing.assert_allclose(output, self.model(np.array([[1, 2, 3]])).data[0, -1], atol=1e-12)

    def test_failed_decode_restores_shared_partial_page_capacity(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=2, dtype=np.float64)
        self.model.decode(1, cache, "source")
        cache.fork("source", "branch")
        original_pages = list(cache.tables["branch"])
        free_before = len(cache.free)
        with mock.patch("rawllm.cache.paged_latent_attention", side_effect=RuntimeError("injected COW error")):
            with self.assertRaisesRegex(RuntimeError, "injected COW error"):
                self.model.decode(2, cache, "branch")
        self.assertEqual(cache.length("branch"), 1)
        self.assertEqual(len(cache.free), free_before)
        self.assertEqual(cache.tables["branch"], original_pages)
        self.assert_cache_invariants(cache)

    def test_late_decode_failure_restores_shared_page_references_and_tables(self):
        from rawllm.cache import paged_latent_attention
        cache = PagedLatentCache(self.config, page_size=2, max_pages=4, dtype=np.float64)
        for token in [1, 2, 3]:
            self.model.decode(token, cache, "source")
        cache.fork("source", "branch")
        cache.fork("source", "sibling")
        original_tables = {name: pages.copy() for name, pages in cache.tables.items()}
        original_lengths = cache.lengths.copy()
        original_refs, original_free = cache.refs.copy(), cache.free.copy()
        original_logical = logical_cache(cache, "source")
        call_count = 0
        def attention_or_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("second layer failed")
            return paged_latent_attention(*args, **kwargs)
        with mock.patch("rawllm.cache.paged_latent_attention", side_effect=attention_or_fail):
            with self.assertRaisesRegex(RuntimeError, "second layer failed"):
                self.model.decode(4, cache, "branch")
        self.assertEqual(cache.tables, original_tables)
        self.assertEqual(cache.lengths, original_lengths)
        self.assertEqual(cache.free, original_free)
        np.testing.assert_equal(cache.refs, original_refs)
        for sequence_id in ["source", "branch", "sibling"]:
            for old_layer, new_layer in zip(original_logical, logical_cache(cache, sequence_id)):
                for old_values, new_values in zip(old_layer, new_layer):
                    np.testing.assert_equal(old_values, new_values)
        self.assert_cache_invariants(cache)
        retried_output = self.model.decode(4, cache, "branch")
        with no_grad():
            expected = self.model(np.array([[1, 2, 3, 4]])).data[0, -1]
        np.testing.assert_allclose(retried_output, expected, atol=1e-12, rtol=1e-12)
        self.assert_cache_invariants(cache)

    def test_reservation_exhaustion_preserves_forks_and_absent_sequence(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=1, dtype=np.float64)
        self.model.decode(1, cache, "source")
        cache.fork("source", "branch")
        tables, lengths = {k: v.copy() for k, v in cache.tables.items()}, cache.lengths.copy()
        refs = cache.refs.copy()
        for sequence_id in ["branch", "new_sequence"]:
            with self.subTest(sequence_id=sequence_id):
                with self.assertRaises(MemoryError):
                    self.model.decode(2, cache, sequence_id)
                self.assertEqual(cache.tables, tables)
                self.assertEqual(cache.lengths, lengths)
                np.testing.assert_equal(cache.refs, refs)
                self.assert_cache_invariants(cache)

    def test_new_sequence_inference_failure_removes_table_and_zeroes_page(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=2, dtype=np.float64)
        original_free = cache.free.copy()
        with mock.patch("rawllm.cache.paged_latent_attention", side_effect=RuntimeError("new sequence failed")):
            with self.assertRaises(RuntimeError):
                self.model.decode(1, cache, "new_sequence")
        self.assertEqual(cache.tables, {})
        self.assertEqual(cache.lengths, {})
        self.assertEqual(cache.free, original_free)
        np.testing.assert_equal(cache.refs, 0)
        np.testing.assert_equal(cache.latent, 0)
        np.testing.assert_equal(cache.rotary, 0)
        self.assert_cache_invariants(cache)

    def test_storage_dtype_guards_run_before_allocation(self):
        for dtype in (np.int8, np.int64, np.bool_, np.float16, np.complex64):
            with self.subTest(dtype=dtype):
                with mock.patch("rawllm.model.np.random.default_rng", side_effect=AssertionError("allocation began")):
                    with self.assertRaises(TypeError):
                        Transformer(self.config, dtype=dtype)
                with mock.patch("rawllm.cache.np.zeros", side_effect=AssertionError("allocation began")):
                    with self.assertRaises(TypeError):
                        PagedLatentCache(self.config, dtype=dtype)

    def test_config_rejects_nonfinite_rotary_base_and_norm_epsilon(self):
        for field in ("rope_base", "norm_eps"):
            for value in (np.nan, np.inf, -np.inf, 0.0, -1.0):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        tiny_config(**{field: value})

    def test_paged_decode_rejects_emulated_precision_before_reservation(self):
        cache = PagedLatentCache(self.config, page_size=2, max_pages=2, dtype=np.float64)
        for precision in ("fp16", "bf16", "unknown"):
            with self.subTest(precision=precision):
                self.model.precision = precision
                with self.assertRaisesRegex(ValueError, "requires fp32"):
                    self.model.decode(1, cache)
                self.assertEqual(cache.tables, {})
                self.assertEqual(cache.lengths, {})
                self.assertEqual(len(cache.free), cache.max_pages)

    def test_packed_attention_blocks_cross_document_leakage(self):
        first = np.array([[1, 2, 3, 4, 5, 6]])
        changed_first_document = np.array([[8, 9, 10, 4, 5, 6]])
        segments = np.array([0, 0, 0, 1, 1, 1])
        mask = (segments[:, None] == segments[None, :])[None, None, :, :]
        with no_grad():
            logits = self.model(first, attention_mask=mask).data
            changed = self.model(changed_first_document, attention_mask=mask).data
            independent_second = self.model(np.array([[4, 5, 6]])).data
            unmasked = self.model(first).data
            unmasked_changed = self.model(changed_first_document).data
        np.testing.assert_equal(logits[:, 3:], changed[:, 3:])
        np.testing.assert_allclose(logits[:, 3:], independent_second, atol=1e-12, rtol=1e-12)
        self.assertGreater(np.max(np.abs(unmasked[:, 3:] - unmasked_changed[:, 3:])), 1e-8)

    def test_future_tokens_cannot_change_prefix_outputs(self):
        with no_grad():
            a = self.model(np.array([[1, 2, 3, 4]])).data
            b = self.model(np.array([[1, 2, 9, 10]])).data
        np.testing.assert_equal(a[:, :2], b[:, :2])

    def test_invalid_tokens_and_sequence_limits_do_not_reserve_cache(self):
        config = tiny_config(max_seq_len=2)
        model = Transformer(config, dtype=np.float64)
        cache = PagedLatentCache(config, page_size=2, max_pages=2, dtype=np.float64)
        with self.assertRaises(ValueError):
            model.decode(config.vocab_size, cache)
        self.assertEqual(cache.length("default"), 0)
        model.decode(1, cache)
        model.decode(2, cache)
        with self.assertRaises(ValueError):
            model.decode(3, cache)
        self.assertEqual(cache.length("default"), 2)
        self.assert_cache_invariants(cache)


if __name__ == "__main__":
    unittest.main()
