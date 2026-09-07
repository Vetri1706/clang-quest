"""A small explicit reverse-mode automatic differentiation engine.

Each operation records its inputs and a vector-Jacobian-product function.
``backward`` walks the graph in reverse topological order, accumulating
contributions from every consumer before visiting an input. Leaf gradients
accumulate across backward calls; call ``zero_grad`` between optimizer steps.
Intermediate gradients are discarded unless ``retain_grad`` is requested.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterable
import numpy as np

from . import kernels


_grad_enabled = ContextVar("rawllm_grad_enabled", default=True)


@contextmanager
def no_grad():
    """Disable graph creation in the current execution context."""
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def _unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    gradient = np.asarray(gradient)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


def _axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalized = []
    for a in axes:
        if not -ndim <= a < ndim:
            raise ValueError("axis is out of bounds")
        normalized.append(a % ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate reduction axes")
    return tuple(normalized)


class Tensor:
    """A NumPy array with a dynamically constructed differentiation graph."""

    __array_priority__ = 1000

    def __init__(self, data, requires_grad: bool = False, name: str | None = None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data)
        if self.data.dtype.kind not in "biuf":
            raise TypeError("Tensor supports real numeric and boolean arrays")
        if requires_grad and self.data.dtype.kind != "f":
            self.data = self.data.astype(np.float64)
        self.requires_grad = bool(requires_grad)
        self.name = name
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward_fn: Callable | None = None
        self._retain_grad = False

    @classmethod
    def _from_op(cls, data, parents: Iterable["Tensor"], backward: Callable):
        """Create an op; backward(upstream) returns one gradient per parent.

        Parent gradients must already have the parent's shape. Return None
        when an input is intentionally nondifferentiable. Callbacks never
        mutate ``.grad`` themselves, so shared inputs and repeated backward
        calls accumulate correctly.
        """
        parents = tuple(parents)
        enabled = _grad_enabled.get() and any(p.requires_grad for p in parents)
        result = Tensor(data, requires_grad=enabled)
        if enabled:
            result._parents = parents
            result._backward_fn = backward
        return result

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    def numel(self):
        return self.size

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad}, name={self.name!r})"

    def item(self):
        return self.data.item()

    def numpy(self, copy: bool = True):
        return self.data.copy() if copy else self.data

    def detach(self, copy: bool = False):
        return Tensor(self.data.copy() if copy else self.data, name=self.name)

    def zero_grad(self, set_to_none: bool = True):
        if set_to_none:
            self.grad = None
        elif self.grad is not None:
            self.grad.fill(0)
        else:
            self.grad = np.zeros_like(self.data, dtype=np.result_type(self.dtype, np.float32))

    def retain_grad(self):
        """Keep this intermediate tensor's gradient after backpropagation."""
        if not self.requires_grad:
            raise RuntimeError("cannot retain a gradient on a nondifferentiable tensor")
        self._retain_grad = True
        return self

    def backward(self, gradient=None):
        """Backpropagate a scalar loss or an explicit output cotangent.

        The iterative topological traversal also handles graphs deeper than
        Python's recursion limit. A fresh per-call gradient map prevents
        previously accumulated intermediate gradients from being propagated
        a second time.
        """
        if not self.requires_grad:
            raise RuntimeError("cannot backpropagate a Tensor that does not require gradients")
        if gradient is None:
            if self.size != 1:
                raise ValueError("non-scalar backward requires an explicit gradient")
            gradient = np.ones_like(self.data)
        gradient = np.asarray(gradient, dtype=np.result_type(self.dtype, np.float32))
        if gradient.shape != self.shape:
            raise ValueError(f"gradient shape {gradient.shape} does not match {self.shape}")
        visited, order = set(), []
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents if p.requires_grad)
        gradients = {self: gradient}
        for node in reversed(order):
            upstream = gradients.pop(node, None)
            if upstream is None:
                continue
            upstream = np.asarray(upstream, dtype=np.result_type(node.dtype, np.float32))
            if not node._parents or node._retain_grad:
                node.grad = upstream.copy() if node.grad is None else node.grad + upstream
            if node._backward_fn is None:
                continue
            parent_gradients = node._backward_fn(upstream)
            if len(parent_gradients) != len(node._parents):
                raise RuntimeError("operation returned the wrong number of parent gradients")
            for parent, g in zip(node._parents, parent_gradients):
                if not parent.requires_grad or g is None:
                    continue
                g = np.asarray(g, dtype=np.result_type(parent.dtype, np.float32))
                if g.shape != parent.shape:
                    raise RuntimeError(f"operation gradient has shape {g.shape}, expected {parent.shape}")
                gradients[parent] = gradients[parent] + g if parent in gradients else g

    def __add__(self, other):
        other = as_tensor(other)
        return self._from_op(self.data + other.data, (self, other),
                             lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, other.shape)))

    __radd__ = __add__

    def __neg__(self):
        return self._from_op(-self.data, (self,), lambda g: (-g,))

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) + (-self)

    def __mul__(self, other):
        other = as_tensor(other)
        return self._from_op(self.data * other.data, (self, other),
            lambda g: (_unbroadcast(g * other.data, self.shape),
                       _unbroadcast(g * self.data, other.shape)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_tensor(other)
        return self._from_op(self.data / other.data, (self, other),
            lambda g: (_unbroadcast(g / other.data, self.shape),
                       _unbroadcast(-g * self.data / other.data**2, other.shape)))

    def __rtruediv__(self, other):
        return as_tensor(other) / self

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            return (exponent * self.log()).exp()
        if not np.isscalar(exponent):
            raise TypeError("power exponent must be a scalar or Tensor")
        out = self.data**exponent
        def backward(g):
            if exponent == 0:
                return (np.zeros_like(self.data),)
            return (g * exponent * self.data ** (exponent - 1),)
        return self._from_op(out, (self,), backward)

    def __matmul__(self, other):
        other = as_tensor(other)
        out = kernels.matmul(self.data, other.data)
        avector, bvector = self.ndim == 1, other.ndim == 1
        a = self.data[None, :] if avector else self.data
        b = other.data[:, None] if bvector else other.data
        def backward(g):
            if avector and bvector:
                g = g.reshape(1, 1)
            elif avector:
                g = np.expand_dims(g, -2)
            elif bvector:
                g = np.expand_dims(g, -1)
            ga = np.matmul(g, b.swapaxes(-1, -2))
            gb = np.matmul(a.swapaxes(-1, -2), g)
            if avector:
                ga = np.squeeze(ga, axis=-2)
            if bvector:
                gb = np.squeeze(gb, axis=-1)
            return _unbroadcast(ga, self.shape), _unbroadcast(gb, other.shape)
        return self._from_op(out, (self, other), backward)

    def __rmatmul__(self, other):
        return as_tensor(other) @ self

    def sum(self, axis=None, keepdims: bool = False):
        axes = _axes(axis, self.ndim)
        out = self.data.sum(axis=axis, keepdims=keepdims)
        def backward(g):
            if not keepdims:
                for a in sorted(axes):
                    g = np.expand_dims(g, a)
            return (np.broadcast_to(g, self.shape),)
        return self._from_op(out, (self,), backward)

    def mean(self, axis=None, keepdims: bool = False):
        axes = _axes(axis, self.ndim)
        divisor = int(np.prod([self.shape[a] for a in axes], dtype=np.int64))
        if divisor == 0:
            raise ValueError("mean over an empty axis is undefined")
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self.data.reshape(shape)
        return self._from_op(out, (self,), lambda g: (g.reshape(self.shape),))

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        if self.ndim == 0:
            if start_dim not in (0, -1) or end_dim not in (0, -1):
                raise ValueError("scalar flatten dimensions must be 0 or -1")
            return self.reshape(1)
        if not -self.ndim <= start_dim < self.ndim or not -self.ndim <= end_dim < self.ndim:
            raise ValueError("flatten dimension is out of bounds")
        start, end = start_dim % self.ndim, end_dim % self.ndim
        if start > end:
            raise ValueError("start_dim must not exceed end_dim")
        return self.reshape((*self.shape[:start], -1, *self.shape[end + 1 :]))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        if len(axes) != self.ndim or any(not -self.ndim <= a < self.ndim for a in axes):
            raise ValueError("transpose axes must be a permutation of all dimensions")
        axes = tuple(a % self.ndim for a in axes)
        out = self.data.transpose(axes)
        inverse = tuple(np.argsort(axes))
        return self._from_op(out, (self,), lambda g: (g.transpose(inverse),))

    def swapaxes(self, axis1, axis2):
        axes = list(range(self.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self.transpose(axes)

    def __getitem__(self, key):
        out = self.data[key]
        def backward(g):
            grad = np.zeros_like(self.data, dtype=np.result_type(self.dtype, np.float32))
            np.add.at(grad, key, g)
            return (grad,)
        return self._from_op(out, (self,), backward)

    def astype(self, dtype):
        if np.dtype(dtype).kind != "f" and self.requires_grad and _grad_enabled.get():
            raise TypeError("a differentiable cast must target a floating-point dtype")
        return self._from_op(self.data.astype(dtype), (self,),
                             lambda g: (g.astype(np.result_type(self.dtype, np.float32)),))

    def exp(self):
        out = np.exp(self.data)
        return self._from_op(out, (self,), lambda g: (g * out,))

    def log(self):
        return self._from_op(np.log(self.data), (self,), lambda g: (g / self.data,))

    def sqrt(self):
        return self**0.5

    def tanh(self):
        out = np.tanh(self.data)
        return self._from_op(out, (self,), lambda g: (g * (1 - out**2),))

    def sigmoid(self):
        work = self.data.astype(np.result_type(self.dtype, np.float32), copy=False)
        z = np.exp(-np.abs(work))
        out = np.where(work >= 0, 1 / (1 + z), z / (1 + z))
        return self._from_op(out, (self,), lambda g: (g * out * (1 - out),))

    def silu(self):
        return self * self.sigmoid()

    def relu(self):
        return self._from_op(np.maximum(self.data, 0), (self,),
                             lambda g: (g * (self.data > 0),))

    def softplus(self):
        return softplus(self)

    def softmax(self, axis=-1):
        return softmax(self, axis)

    def log_softmax(self, axis=-1):
        return log_softmax(self, axis)

    def clip(self, minimum, maximum):
        out = np.clip(self.data, minimum, maximum)
        inside = (self.data >= minimum) & (self.data <= maximum)
        return self._from_op(out, (self,), lambda g: (g * inside,))


class Parameter(Tensor):
    """A trainable leaf tensor."""

    def __init__(self, data, name: str | None = None):
        super().__init__(data, requires_grad=True, name=name)


def as_tensor(value) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def concatenate(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    tensors = tuple(as_tensor(x) for x in tensors)
    if not tensors:
        raise ValueError("concatenate requires at least one tensor")
    boundaries = np.cumsum([x.shape[axis] for x in tensors[:-1]])
    return Tensor._from_op(np.concatenate([x.data for x in tensors], axis=axis),
                           tensors, lambda g: tuple(np.split(g, boundaries, axis=axis)))


def stack(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    tensors = tuple(as_tensor(x) for x in tensors)
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    return Tensor._from_op(np.stack([x.data for x in tensors], axis=axis), tensors,
                           lambda g: tuple(np.take(g, i, axis=axis) for i in range(len(tensors))))


def embedding(weight: Tensor, indices: np.ndarray) -> Tensor:
    weight = as_tensor(weight)
    indices = np.asarray(indices)
    if weight.ndim != 2 or indices.dtype.kind not in "iu":
        raise ValueError("embedding needs a rank-two weight and integer indices")
    if np.any(indices < 0) or np.any(indices >= weight.shape[0]):
        raise IndexError("embedding index is outside the vocabulary")
    return weight[indices]


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    x = as_tensor(x)
    out = kernels.softmax(x.data, axis=axis)
    return Tensor._from_op(out, (x,),
        lambda g: (out * (g - (g * out).sum(axis=axis, keepdims=True)),))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    x = as_tensor(x)
    out = kernels.log_softmax(x.data, axis=axis)
    p = np.exp(out)
    valid = p.sum(axis=axis, keepdims=True) > 0
    return Tensor._from_op(out, (x,),
        lambda g: (np.where(valid, g - p * g.sum(axis=axis, keepdims=True), 0),))


def logsumexp(x: Tensor, axis=-1, keepdims: bool = False) -> Tensor:
    x = as_tensor(x)
    axes = _axes(axis, x.ndim)
    m = np.max(x.data, axis=axis, keepdims=True)
    shifted = np.full_like(x.data, -np.inf, dtype=np.result_type(x.dtype, np.float32))
    np.subtract(x.data, m, out=shifted, where=np.isfinite(x.data) & np.isfinite(m))
    e = np.exp(shifted)
    total = e.sum(axis=axis, keepdims=True)
    logtotal = np.zeros_like(total)
    np.log(total, out=logtotal, where=total > 0)
    out = m + logtotal
    p = kernels.softmax(x.data, axis=axis)
    def backward(g):
        if not keepdims:
            for a in sorted(axes):
                g = np.expand_dims(g, a)
        return (g * p,)
    if not keepdims:
        out = np.squeeze(out, axis=axes)
    return Tensor._from_op(out, (x,), backward)


def softplus(x: Tensor) -> Tensor:
    x = as_tensor(x)
    out = np.maximum(x.data, 0) + np.log1p(np.exp(-np.abs(x.data)))
    z = np.exp(-np.abs(x.data))
    derivative = np.where(x.data >= 0, 1 / (1 + z), z / (1 + z))
    return Tensor._from_op(out, (x,), lambda g: (g * derivative,))


def maximum(x: Tensor, y: Tensor) -> Tensor:
    """Elementwise maximum; equal operands split the subgradient evenly."""
    x, y = as_tensor(x), as_tensor(y)
    dx = (x.data > y.data) + 0.5 * (x.data == y.data)
    dy = 1 - dx
    return Tensor._from_op(np.maximum(x.data, y.data), (x, y),
        lambda g: (_unbroadcast(g * dx, x.shape), _unbroadcast(g * dy, y.shape)))


def minimum(x: Tensor, y: Tensor) -> Tensor:
    return -maximum(-as_tensor(x), -as_tensor(y))


def cross_entropy(logits: Tensor, targets: np.ndarray, mask: np.ndarray | None = None,
                  reduction: str = "mean", ignore_index: int = -100) -> Tensor:
    """Stable token cross entropy, normalized by the sum of unmasked weights.

    A completely masked batch contributes zero loss and zero gradients.
    ``ignore_index`` is excluded independently of the supplied mask. A float
    mask supplies nonnegative per-token weights, and ``reduction='none'``
    returns weighted losses with the same shape as the targets.
    """
    logits = as_tensor(logits)
    targets = np.asarray(targets)
    if logits.ndim < 1 or targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits except for the vocabulary axis")
    if targets.dtype.kind not in "iu":
        raise TypeError("cross entropy targets must be integer token IDs")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be mean, sum, or none")
    valid = targets != ignore_index
    if np.any(valid & ((targets < 0) | (targets >= logits.shape[-1]))):
        raise IndexError("cross entropy target outside vocabulary")
    weights = valid.astype(np.result_type(logits.dtype, np.float32))
    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask), targets.shape)
        if np.any(mask < 0) or not np.all(np.isfinite(mask)):
            raise ValueError("loss mask must contain finite nonnegative weights")
        weights = weights * mask
    logp = kernels.log_softmax(logits.data, axis=-1)
    safe_targets = np.where(valid, targets, 0)
    selected = np.take_along_axis(logp, safe_targets[..., None], axis=-1)[..., 0]
    losses = np.zeros_like(selected)
    np.multiply(-selected, weights, out=losses, where=weights != 0)
    denominator = float(weights.sum())
    if denominator == 0:
        denominator = 1.0
    out = losses if reduction == "none" else losses.sum()
    if reduction == "mean":
        out = out / denominator
    def backward(g):
        grad = np.exp(logp)
        flat = grad.reshape(-1, grad.shape[-1])
        flat[np.arange(safe_targets.size), safe_targets.reshape(-1)] -= 1
        grad *= weights[..., None]
        if reduction == "mean":
            grad /= denominator
        grad *= g[..., None] if reduction == "none" else g
        return (grad,)
    return Tensor._from_op(out, (logits,), backward)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    x, weight = as_tensor(x), as_tensor(weight)
    if x.ndim < 1 or weight.shape != (x.shape[-1],) or eps <= 0:
        raise ValueError("RMSNorm needs a last-axis weight and positive epsilon")
    work = x.data.astype(np.result_type(x.dtype, np.float32), copy=False)
    inv = 1 / np.sqrt(np.mean(work**2, axis=-1, keepdims=True) + eps)
    normalized = work * inv
    out = normalized * weight.data
    def backward(g):
        dy = g * weight.data
        dx = inv * dy - work * inv**3 * np.mean(dy * work, axis=-1, keepdims=True)
        return dx, _unbroadcast(g * normalized, weight.shape)
    return Tensor._from_op(out, (x, weight), backward)


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    x, weight, bias = as_tensor(x), as_tensor(weight), as_tensor(bias)
    if x.ndim < 1 or weight.shape != (x.shape[-1],) or bias.shape != weight.shape or eps <= 0:
        raise ValueError("LayerNorm needs last-axis weight/bias and positive epsilon")
    work = x.data.astype(np.result_type(x.dtype, np.float32), copy=False)
    centered = work - work.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + eps)
    normalized = centered * inv
    out = normalized * weight.data + bias.data
    def backward(g):
        dy = g * weight.data
        dx = inv * (dy - dy.mean(axis=-1, keepdims=True)
                    - normalized * (dy * normalized).mean(axis=-1, keepdims=True))
        return dx, _unbroadcast(g * normalized, weight.shape), _unbroadcast(g, bias.shape)
    return Tensor._from_op(out, (x, weight, bias), backward)


def rope(x: Tensor, positions: np.ndarray, base: float = 10000.0) -> Tensor:
    """Apply interleaved-pair rotary embeddings along the last feature axis.

    For [B,H,T,D] inputs, positions may be [T], [B,T], or broadcastable to
    [B,H,T]. The feature dimension must be even. No learned parameter is
    introduced by the rotation.
    """
    x = as_tensor(x)
    if x.ndim < 2 or x.shape[-1] % 2 or base <= 0 or not np.isfinite(base):
        raise ValueError("RoPE requires an even feature width, token axis, and positive finite base")
    positions = np.asarray(positions)
    if x.ndim == 4 and positions.ndim == 2 and positions.shape == (x.shape[0], x.shape[2]):
        positions = positions[:, None, :]
    positions = np.broadcast_to(positions, x.shape[:-1])
    dtype = np.result_type(x.dtype, np.float32)
    frequencies = base ** (-np.arange(0, x.shape[-1], 2, dtype=dtype) / x.shape[-1])
    angle = positions[..., None] * frequencies
    cosine, sine = np.cos(angle).astype(dtype), np.sin(angle).astype(dtype)
    even, odd = x.data[..., 0::2], x.data[..., 1::2]
    out = np.empty_like(x.data, dtype=dtype)
    out[..., 0::2] = even * cosine - odd * sine
    out[..., 1::2] = even * sine + odd * cosine
    def backward(g):
        grad = np.empty_like(g)
        grad[..., 0::2] = g[..., 0::2] * cosine + g[..., 1::2] * sine
        grad[..., 1::2] = -g[..., 0::2] * sine + g[..., 1::2] * cosine
        return (grad,)
    return Tensor._from_op(out, (x,), backward)


def attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True,
              mask: np.ndarray | None = None, query_offset: int = 0,
              block_size: int = 32, scale: float | None = None) -> Tensor:
    """Memory-bounded exact attention with explicitly recomputed gradients."""
    q, k, v = as_tensor(q), as_tensor(k), as_tensor(v)
    kwargs = dict(causal=causal, mask=mask, query_offset=query_offset,
                  block_size=block_size, scale=scale)
    out, lse = kernels.attention_forward(q.data, k.data, v.data, **kwargs)
    return Tensor._from_op(out, (q, k, v), lambda g:
        kernels.attention_backward(q.data, k.data, v.data, out, lse, g, **kwargs))
