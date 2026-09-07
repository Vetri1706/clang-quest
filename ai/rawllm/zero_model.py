"""Automatic operation-level ZeRO-3 rematerialization for the MLA Transformer.

No autograd callback captures a full parameter array. Linear and weighted
normalization operators gather one weight for forward, release it, and gather
it again only while computing its backward. Embedding gradients and tied output
projection gradients accumulate into the same sharded parameter. Activations,
one full weight/gradient, collective buffers, and local optimizer temporaries
remain real memory costs; this module does not claim activation checkpointing.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np

from .distributed import ProcessGroup
from .model import Config
from .tensor import Tensor, attention, concatenate, rope
from .zero import ShardedAdamW
from .zero import shard_bounds


def initialized_parameters(config: Config, seed: int):
    """Match Transformer initialization while yielding one named tensor at a time.

    NumPy's Gaussian generator temporarily creates its FP64 output before the
    FP32 cast. Initialization therefore peaks at a constant number of buffers
    of the largest parameter, in addition to already constructed local shards.
    It never constructs a dictionary of the complete model's arrays.
    """
    rng = np.random.default_rng(seed)
    for name, shape in config.shapes().items():
        if len(shape) == 1:
            yield name, np.ones(shape, dtype=np.float32)
        else:
            yield name, rng.normal(0, 0.02, shape).astype(np.float32)


class ZeroTransformer:
    """An MLA Transformer whose only persistent parameter storage is sharded.

    All data ranks must execute the same graph and operation order. Token values
    and masks may differ, but conditional skipping of a module is unsupported.
    This supports the Config architecture, causal/block masks, tied or untied
    embeddings, and FP32 computation. Pipeline/tensor parallel composition and
    mixed-precision emulation are separate features, not integrated here.
    """

    def __init__(self, config: Config, group: ProcessGroup, seed: int = 0,
                 learning_rate: float = 3e-4, weight_decay: float = 0.01,
                 max_parameter_bytes: int = 128 * 1024**2,
                 max_persistent_bytes: int = 128 * 1024**2):
        largest = max(math.prod(shape) * 4 for shape in config.shapes().values())
        if largest > max_parameter_bytes:
            raise MemoryError("largest materialized parameter exceeds the configured local bound")
        local_count = 0
        for shape in config.shapes().values():
            start, stop = shard_bounds(math.prod(shape), group.group_rank, group.size)
            local_count += stop - start
        if local_count * 16 > max_persistent_bytes:
            raise MemoryError("persistent parameter/gradient/Adam shards exceed the configured local bound")
        self.config, self.group = config, group
        self.optimizer = ShardedAdamW(initialized_parameters(config, seed), group,
                                     learning_rate=learning_rate, weight_decay=weight_decay)
        # A single scalar leaf makes parameter-dependent operations
        # differentiable without allocating a Tensor/gradient for every weight.
        self._anchor = Tensor(np.array(0, dtype=np.float32), requires_grad=True)
        self.backward_parameter_calls = 0

    @contextmanager
    def _weight(self, name: str):
        value = self.optimizer.materialize(name)
        try:
            yield value
        finally:
            self.optimizer.release(name)

    def _check_version(self, version: int) -> None:
        if self.optimizer._mutation_generation != version:
            raise RuntimeError("cannot backpropagate a graph after its parameter shards were updated")

    def _record_gradient(self, name: str, gradient: np.ndarray) -> None:
        self.optimizer.accumulate_gradient(name, gradient)
        self.backward_parameter_calls += 1

    def linear(self, x: Tensor, name: str, transpose_weight: bool = False) -> Tensor:
        shape = self.optimizer.partitions[name].shape
        if len(shape) != 2 or x.ndim < 2 or x.shape[-1] != shape[int(transpose_weight)]:
            raise ValueError("linear input and sharded matrix shapes do not agree")
        with self._weight(name) as weight:
            output = x.data @ (weight.T if transpose_weight else weight)
        version = self.optimizer._mutation_generation

        def backward(gradient):
            self._check_version(version)
            with self._weight(name) as weight:
                effective_weight = weight.T if transpose_weight else weight
                input_gradient = gradient @ effective_weight.T
                # Contract all sample axes before matmul. Batched matmul here
                # would allocate one complete weight gradient per batch entry.
                weight_gradient = x.data.reshape(-1, x.shape[-1]).T @ gradient.reshape(-1, gradient.shape[-1])
                if transpose_weight:
                    weight_gradient = weight_gradient.T
                self._record_gradient(name, weight_gradient)
            return input_gradient, np.zeros_like(self._anchor.data)

        return Tensor._from_op(output, (x, self._anchor), backward)

    def norm(self, x: Tensor, name: str) -> Tensor:
        if self.optimizer.partitions[name].shape != (x.shape[-1],):
            raise ValueError("normalization width and scale shape do not agree")
        # These are activation arrays, not full parameter references.
        work = x.data.astype(np.float32, copy=False)
        inverse = 1 / np.sqrt(np.mean(work**2, axis=-1, keepdims=True) + self.config.norm_eps)
        normalized = work * inverse
        with self._weight(name) as weight:
            output = normalized * weight
        version = self.optimizer._mutation_generation

        def backward(gradient):
            self._check_version(version)
            with self._weight(name) as weight:
                dy = gradient * weight
                input_gradient = inverse * dy - work * inverse**3 * np.mean(dy * work, axis=-1, keepdims=True)
                scale_gradient = gradient * normalized
                while scale_gradient.ndim > 1:
                    scale_gradient = scale_gradient.sum(axis=0)
                self._record_gradient(name, scale_gradient)
            return input_gradient, np.zeros_like(self._anchor.data)

        return Tensor._from_op(output, (x, self._anchor), backward)

    def lookup(self, token_ids: np.ndarray) -> Tensor:
        indices = np.asarray(token_ids).copy()
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= self.config.vocab_size):
            raise ValueError("embedding indices must be valid integer token IDs")
        with self._weight("embedding") as weight:
            output = weight[indices]
        version = self.optimizer._mutation_generation

        def backward(gradient):
            self._check_version(version)
            # Dense transient embedding gradient is bounded by the largest
            # full parameter. It is reduce-scattered immediately afterwards.
            weight_gradient = np.zeros(self.optimizer.partitions["embedding"].shape, dtype=np.float32)
            np.add.at(weight_gradient, indices, gradient)
            self._record_gradient("embedding", weight_gradient)
            return (np.zeros_like(self._anchor.data),)

        return Tensor._from_op(output, (self._anchor,), backward)

    def block(self, x: Tensor, index: int, mask: np.ndarray | None = None) -> Tensor:
        c = self.config
        prefix = f"blocks.{index}."
        project = lambda value, name: self.linear(value, prefix + name)
        batch, length, _ = x.shape
        positions = np.arange(length)
        u = self.norm(x, prefix + "attn_norm")
        query_latent = self.norm(project(u, "q_down"), prefix + "q_norm")
        query = project(query_latent, "q_up").reshape(batch, length, c.heads, c.content_dim + c.rope_dim).transpose(0, 2, 1, 3)
        query_content = query[..., :c.content_dim]
        query_rotary = rope(query[..., c.content_dim:], positions, c.rope_base)
        latent = self.norm(project(u, "kv_down"), prefix + "kv_norm")
        key_content = project(latent, "k_up").reshape(batch, length, c.heads, c.content_dim).transpose(0, 2, 1, 3)
        key_rotary = rope(project(u, "k_rope").reshape(batch, 1, length, c.rope_dim), positions, c.rope_base)
        key_rotary = key_rotary * np.ones((1, c.heads, 1, 1), dtype=x.dtype)
        values = project(latent, "v_up").reshape(batch, length, c.heads, c.value_dim).transpose(0, 2, 1, 3)
        attended = attention(concatenate((query_content, query_rotary), axis=-1),
                             concatenate((key_content, key_rotary), axis=-1), values,
                             causal=True, mask=mask, block_size=32)
        x = x + project(attended.transpose(0, 2, 1, 3).reshape(batch, length, c.heads * c.value_dim), "o")
        u = self.norm(x, prefix + "ffn_norm")
        return x + project(project(u, "gate").silu() * project(u, "up"), "down")

    def __call__(self, token_ids: np.ndarray, attention_mask: np.ndarray | None = None) -> Tensor:
        ids = np.asarray(token_ids)
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.max_seq_len:
            raise ValueError("token IDs must be [batch, time] within the context length")
        x = self.lookup(ids)
        for index in range(self.config.layers):
            x = self.block(x, index, attention_mask)
        x = self.norm(x, "final_norm")
        return self.linear(x, "embedding", transpose_weight=True) if self.config.tie_embeddings else self.linear(x, "head")

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
        self._anchor.zero_grad()

    def step(self, gradient_scale: float = 1.0, max_grad_norm: float | None = None) -> dict:
        report = self.optimizer.step(gradient_scale=gradient_scale, max_grad_norm=max_grad_norm)
        self._anchor.zero_grad()
        return report

    def memory_report(self) -> dict:
        optimizer = self.optimizer
        return {"global_parameters": self.config.parameter_count,
                "locally_owned_parameters": optimizer.parameters.size,
                "persistent_parameter_gradient_adam_bytes": optimizer.persistent_bytes,
                "largest_initial_tensor_bytes": optimizer.largest_initial_tensor_bytes,
                "largest_materialized_tensor_bytes": optimizer.peak_materialized_bytes,
                "maximum_simultaneous_materialized_tensors": optimizer.maximum_simultaneous_materializations,
                "materialization_count": optimizer.materialization_count,
                "backward_parameter_calls": self.backward_parameter_calls,
                "active_materialization": optimizer._active_name,
                "memory_exclusions": "saved activations, full transient gradient, received shard-list/serialization/coordinator buffers, local Adam arithmetic temporaries, and initialization temporaries"}
