"""Finite-difference and numerical-equivalence checks for the tensor engine."""

import unittest
import numpy as np

from rawllm import kernels
from rawllm.tensor import (
    Tensor, Parameter, attention, concatenate, stack, embedding,
    softmax, log_softmax, logsumexp, cross_entropy, rms_norm, layer_norm,
    rope, softplus, no_grad,
)


def check_gradients(test, fn, values, *, atol=2e-6, rtol=2e-5):
    """Compare each analytical input derivative to centered finite differences."""
    tensors = [Tensor(np.array(a, dtype=np.float64), requires_grad=True) for a in values]
    loss = fn(*tensors)
    loss.backward()
    analytical = [t.grad.copy() for t in tensors]
    epsilon = 1e-6
    for tensor, expected in zip(tensors, analytical):
        numerical = np.zeros_like(tensor.data)
        for index in np.ndindex(tensor.shape):
            original = tensor.data[index]
            tensor.data[index] = original + epsilon
            plus = float(fn(*tensors).data)
            tensor.data[index] = original - epsilon
            minus = float(fn(*tensors).data)
            tensor.data[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(expected, numerical, atol=atol, rtol=rtol)


class TensorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(71)

    def test_broadcast_arithmetic_gradients(self):
        a = self.rng.uniform(0.5, 1.5, (2, 3))
        b = self.rng.uniform(0.5, 1.5, (1, 3))
        check_gradients(self, lambda x, y: ((x * y + 0.3) / (y + 1) - x**2).sum(), [a, b])

    def test_scalar_broadcast_gradient(self):
        check_gradients(self, lambda x, y: ((x + y) * y).mean(),
                        [self.rng.normal(size=(2, 3)), np.array(0.7)])

    def test_accumulation_on_shared_graph_and_repeated_backward(self):
        x = Parameter(np.array([1.0, 2.0]))
        shared = x * x
        loss = (shared + shared * 3).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, [8, 16])
        loss.backward()
        np.testing.assert_allclose(x.grad, [16, 32])
        x.zero_grad()
        self.assertIsNone(x.grad)
        loss.backward()
        np.testing.assert_allclose(x.grad, [8, 16])

    def test_backward_is_iterative_for_deep_graph(self):
        x = Parameter(np.array(1.0))
        y = x
        for _ in range(1500):
            y = y + 0.01
        y.backward()
        self.assertEqual(x.grad, 1)

    def test_explicit_cotangent(self):
        x = Parameter(np.array([1.0, 2.0]))
        (x * x).backward(np.array([3.0, 4.0]))
        np.testing.assert_allclose(x.grad, [6, 16])
        with self.assertRaises(ValueError):
            (x * x).backward()

    def test_matmul_gradient_shapes(self):
        shapes = [((3,), (3,)), ((3,), (3, 2)), ((2, 3), (3,)),
                  ((2, 3), (3, 2)), ((2, 1, 2, 3), (1, 3, 3, 2)),
                  ((3,), (2, 3, 2)), ((2, 2, 3), (3,))]
        for ashape, bshape in shapes:
            with self.subTest(a=ashape, b=bshape):
                a, b = self.rng.normal(size=ashape), self.rng.normal(size=bshape)
                check_gradients(self, lambda x, y: ((x @ y)**2).sum(), [a, b])

    def test_tiled_matmul_matches_numpy_with_partial_tiles(self):
        a, b = self.rng.normal(size=(7, 11)), self.rng.normal(size=(11, 5))
        np.testing.assert_allclose(kernels.matmul_tiled(a, b, block_size=3), a @ b,
                                   atol=1e-13, rtol=1e-13)

    def test_reductions_shape_transforms(self):
        a = self.rng.normal(size=(2, 3, 4))
        check_gradients(self, lambda x: ((x.sum(axis=(0, -1))**2).mean()
                                        + x.mean(axis=1, keepdims=True).sum()), [a])
        check_gradients(self, lambda x: x.transpose(2, 0, 1).reshape(4, 6).swapaxes(0, 1).sum(), [a])

    def test_embedding_repeated_ids_and_slicing(self):
        weight = self.rng.normal(size=(4, 3))
        ids = np.array([[0, 1, 1], [3, 0, 1]])
        check_gradients(self, lambda w: (embedding(w, ids)**2).sum(), [weight])
        w = Parameter(weight)
        embedding(w, ids).sum().backward()
        np.testing.assert_equal(w.grad[:, 0], [2, 3, 0, 1])
        check_gradients(self, lambda x: x[::-1, 1:].tanh().sum(), [weight])

    def test_concatenate_and_stack(self):
        a, b = self.rng.normal(size=(2, 3)), self.rng.normal(size=(2, 3))
        check_gradients(self, lambda x, y: (concatenate([x, y, x], axis=1)**2).sum(), [a, b])
        check_gradients(self, lambda x, y: (stack([x, y, x], axis=-1)**2).sum(), [a, b])

    def test_elementwise_activations(self):
        x = self.rng.uniform(0.2, 2.0, (2, 3))
        for operation in (lambda t: t.exp(), lambda t: t.log(), lambda t: t.tanh(),
                          lambda t: t.sigmoid(), lambda t: t.silu(), softplus):
            with self.subTest(operation=operation):
                check_gradients(self, lambda t: operation(t).sum(), [x])

    def test_stable_extreme_activations(self):
        x = Tensor(np.array([-1000.0, 0.0, 1000.0]), requires_grad=True)
        y = x.sigmoid() + softplus(x)
        self.assertTrue(np.isfinite(y.data).all())
        y.sum().backward()
        self.assertTrue(np.isfinite(x.grad).all())

    def test_softmax_and_log_softmax_gradients(self):
        x, weights = self.rng.normal(size=(2, 3)), self.rng.normal(size=(2, 3))
        check_gradients(self, lambda t: (softmax(t) * weights).sum(), [x])
        check_gradients(self, lambda t: (log_softmax(t) * weights).sum(), [x])
        check_gradients(self, lambda t: logsumexp(t, axis=(0, 1)).sum(), [x])
        check_gradients(self, lambda t: logsumexp(t, axis=0, keepdims=True).sum(), [x])
        np.testing.assert_allclose(np.exp(kernels.log_softmax(np.array([1000., -1000.]))), [1, 0])
        self.assertEqual(kernels.log_softmax(np.array([1000., -1000.]))[1], -2000)

    def test_softmax_entirely_masked_rows(self):
        x = Parameter(np.array([[-np.inf, -np.inf], [0., -np.inf]]))
        softmax(x).sum().backward()
        np.testing.assert_equal(x.grad, 0)
        np.testing.assert_equal(kernels.softmax(x.data), [[0, 0], [1, 0]])

    def test_cross_entropy_with_weighted_mask(self):
        x = self.rng.normal(size=(2, 3, 4))
        targets = np.array([[0, 1, 2], [3, -100, 1]])
        mask = np.array([[1.0, 0.0, 0.5], [1.0, 1.0, 1.0]])
        for reduction in ("mean", "sum", "none"):
            with self.subTest(reduction=reduction):
                check_gradients(self,
                    lambda t: cross_entropy(t, targets, mask, reduction=reduction).sum(), [x])

    def test_cross_entropy_all_masked_float32(self):
        x = Parameter(np.ones((2, 3), dtype=np.float32))
        loss = cross_entropy(x, np.array([-100, -100]))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        np.testing.assert_equal(x.grad, 0)

    def test_rms_norm_gradients(self):
        x, w = self.rng.normal(size=(2, 3, 4)), self.rng.normal(size=4)
        upstream = self.rng.normal(size=x.shape)
        check_gradients(self, lambda a, b: (rms_norm(a, b) * upstream).sum(), [x, w])

    def test_layer_norm_gradients(self):
        x, w, b = self.rng.normal(size=(2, 3, 4)), self.rng.normal(size=4), self.rng.normal(size=4)
        upstream = self.rng.normal(size=x.shape)
        check_gradients(self, lambda a, b, c: (layer_norm(a, b, c) * upstream).sum(), [x, w, b])

    def test_rope_gradients_and_norm(self):
        x = self.rng.normal(size=(2, 2, 3, 4))
        upstream = self.rng.normal(size=x.shape)
        positions = np.array([0, 3, 11])
        check_gradients(self, lambda t: (rope(t, positions) * upstream).sum(), [x])
        rotated = rope(Tensor(x), positions).data
        np.testing.assert_allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1))
        np.testing.assert_equal(rotated[:, :, 0], x[:, :, 0])

    def test_rope_batch_positions(self):
        x = self.rng.normal(size=(2, 2, 3, 4))
        positions = np.array([[0, 1, 2], [5, 6, 7]])
        all_rotated = rope(Tensor(x), positions).data
        for batch in range(2):
            reference = rope(Tensor(x[batch : batch + 1]), positions[batch]).data
            np.testing.assert_allclose(all_rotated[batch : batch + 1], reference)

    def test_no_grad_and_detach(self):
        p = Parameter(np.ones((2, 3)))
        with no_grad():
            y = p * p
            self.assertFalse(y.requires_grad)
            self.assertEqual(y._parents, ())
        self.assertTrue((p * p).requires_grad)
        self.assertFalse(p.detach().requires_grad)

    def test_retaining_intermediate_gradients(self):
        p = Parameter(np.array([1.0, 2.0]))
        hidden = p * 2
        hidden.sum().backward()
        self.assertIsNone(hidden.grad)
        hidden.retain_grad()
        hidden.sum().backward()
        np.testing.assert_equal(hidden.grad, 1)
        np.testing.assert_equal(p.grad, 4)

    def test_scalar_reshape_and_flatten(self):
        p = Parameter(np.array([2.0]))
        scalar = p.reshape(())
        self.assertEqual(scalar.shape, ())
        scalar.flatten().sum().backward()
        np.testing.assert_equal(p.grad, 1)
        with self.assertRaises(ValueError):
            p.transpose(3)

    def test_cast_gradient_accumulates_in_float32(self):
        p = Parameter(np.ones((2, 3), dtype=np.float16))
        p.astype(np.float32).sum().backward()
        self.assertEqual(p.grad.dtype, np.float32)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(89)
        self.q = self.rng.normal(size=(1, 2, 3, 2))
        self.k = self.rng.normal(size=(1, 2, 4, 2))
        self.v = self.rng.normal(size=(1, 2, 4, 3))
        self.upstream = self.rng.normal(size=(1, 2, 3, 3))

    def dense_reference(self, causal, mask=None, offset=0):
        scores = self.q @ self.k.swapaxes(-1, -2) / np.sqrt(self.q.shape[-1])
        if causal:
            allowed = np.arange(self.k.shape[2])[None, :] <= (np.arange(self.q.shape[2]) + offset)[:, None]
            scores = np.where(allowed, scores, -np.inf)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        return kernels.softmax(scores) @ self.v

    def test_forward_matches_dense_multiple_tiles(self):
        for causal in (False, True):
            for offset in (0, 1):
                for block_size in (1, 2, 3, 8):
                    with self.subTest(causal=causal, offset=offset, block_size=block_size):
                        result = attention(Tensor(self.q), Tensor(self.k), Tensor(self.v),
                                           causal=causal, query_offset=offset, block_size=block_size)
                        np.testing.assert_allclose(result.data, self.dense_reference(causal, offset=offset),
                                                   rtol=1e-13, atol=1e-13)

    def test_noncausal_attention_gradients(self):
        check_gradients(self, lambda q, k, v: (attention(q, k, v, causal=False, block_size=2)
                                              * self.upstream).sum(), [self.q, self.k, self.v])

    def test_causal_attention_gradients_with_query_offset(self):
        check_gradients(self, lambda q, k, v: (attention(q, k, v, causal=True,
                                             query_offset=1, block_size=2) * self.upstream).sum(),
                        [self.q, self.k, self.v])

    def test_masked_attention_gradients_and_entirely_masked_rows(self):
        mask = np.array([[[[True, False, True, False],
                           [False, False, False, False],
                           [True, True, False, True]]]])
        fn = lambda q, k, v: (attention(q, k, v, causal=False, mask=mask, block_size=2)
                              * self.upstream).sum()
        check_gradients(self, fn, [self.q, self.k, self.v])
        result = attention(Tensor(self.q), Tensor(self.k), Tensor(self.v), causal=False, mask=mask, block_size=2)
        np.testing.assert_allclose(result.data, self.dense_reference(False, mask=mask), rtol=1e-13, atol=1e-13)
        np.testing.assert_equal(result.data[:, :, 1], 0)

    def test_empty_key_sequence(self):
        q, k, v = Parameter(self.q), Parameter(self.k[:, :, :0]), Parameter(self.v[:, :, :0])
        out = attention(q, k, v, block_size=2)
        out.sum().backward()
        np.testing.assert_equal(out.data, 0)
        np.testing.assert_equal(q.grad, 0)
        self.assertEqual(k.grad.size, 0)
        self.assertEqual(v.grad.size, 0)

    def test_large_logits_remain_finite(self):
        q, k = Parameter(self.q * 100), Parameter(self.k * 100)
        v = Parameter(self.v)
        out = attention(q, k, v, causal=True, block_size=2)
        out.sum().backward()
        for a in (out.data, q.grad, k.grad, v.grad):
            self.assertTrue(np.isfinite(a).all())


if __name__ == "__main__":
    unittest.main()
