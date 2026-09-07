"""Explicit ZeRO Stage 3 parameter/gradient/Adam-state partitioning on NumPy.

Each tensor is split independently (including uneven and empty pieces), and its
local pieces occupy one flat FP32 allocation. Materialization gathers only one
named tensor at a time. The caller must discard its own full tensors/gradients
after use; Python references cannot be revoked by an optimizer. The coordinator
still receives full collective traffic and is not itself memory sharded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np

from .distributed import ProcessGroup


@dataclass(frozen=True)
class TensorPartition:
    name: str
    shape: tuple[int, ...]
    total_size: int
    global_start: int
    global_stop: int
    local_start: int
    local_stop: int


def shard_bounds(size: int, rank: int, world_size: int) -> tuple[int, int]:
    if size < 0 or not 0 <= rank < world_size:
        raise ValueError("invalid shard coordinates")
    base, remainder = divmod(size, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


class ShardedAdamW:
    """Memory-sharded AdamW with explicit tensor gather/release scheduling.

    Initial arrays are read and local slices are copied; no reference to those
    input arrays is retained. An iterable of (name, array) pairs can stream one
    tensor at a time, avoiding construction of a complete initial model. Persistent memory
    is 16 bytes per locally owned parameter (weight, gradient, first and second
    moments), excluding metadata; optimizer arithmetic has local-sized scratch.

    Use ``materialize(name)`` before that layer's forward/backward, call
    ``accumulate_gradient(name, gradient)`` after backward, and ``release(name)``
    before materializing another name. This is a scheduling primitive; it does
    not automatically replace model parameters or rebuild autograd graphs.
    """

    def __init__(self, parameters: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]], group: ProcessGroup,
                 learning_rate: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 epsilon: float = 1e-8, weight_decay: float = 0.01,
                 gradient_reduction: str = "mean"):
        if not parameters or learning_rate < 0 or epsilon <= 0 or weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if len(betas) != 2 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or gradient_reduction not in {"sum", "mean"}:
            raise ValueError("invalid optimizer betas or reduction")
        scalars = (learning_rate, epsilon, weight_decay, *betas)
        if not all(np.isfinite(x) for x in scalars):
            raise ValueError("optimizer settings must be finite")
        self.group = group
        self.learning_rate, self.betas = learning_rate, betas
        self.epsilon, self.weight_decay = epsilon, weight_decay
        self.gradient_reduction = gradient_reduction
        self.partitions: dict[str, TensorPartition] = {}
        pieces, local_offset = [], 0
        self.largest_initial_tensor_bytes = 0
        items = parameters.items() if isinstance(parameters, Mapping) else parameters
        for name, value in items:
            array = np.asarray(value, dtype=np.float32)
            if not isinstance(name, str) or not name or name in self.partitions or not np.isfinite(array).all():
                raise ValueError("parameter names and initial values must be valid")
            self.largest_initial_tensor_bytes = max(self.largest_initial_tensor_bytes, array.nbytes)
            start, stop = shard_bounds(array.size, group.group_rank, group.size)
            self.partitions[name] = TensorPartition(name, array.shape, array.size, start, stop,
                                                   local_offset, local_offset + stop - start)
            pieces.append(array.reshape(-1)[start:stop].copy())
            local_offset += stop - start
            del array, value
        if not pieces:
            raise ValueError("optimizer needs at least one parameter")
        self.parameters = np.concatenate(pieces).astype(np.float32, copy=False)
        del pieces
        layout = {"parameters": [(p.name, p.shape) for p in self.partitions.values()],
                  "learning_rate": learning_rate, "betas": betas, "epsilon": epsilon,
                  "weight_decay": weight_decay, "gradient_reduction": gradient_reduction}
        signature = np.frombuffer(hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).digest(), dtype=np.uint8)
        if any(not np.array_equal(signature, other) for other in group.all_gather(signature)):
            raise ValueError("optimizer ranks disagree about parameter layout or hyperparameters")
        self.gradients = np.zeros_like(self.parameters)
        self.first_moment = np.zeros_like(self.parameters)
        self.second_moment = np.zeros_like(self.parameters)
        self.step_number = 0
        self._mutation_generation = 0
        self._accumulations = {name: 0 for name in self.partitions}
        self._active_name: str | None = None
        self._active_value: np.ndarray | None = None
        self.materialization_count = 0
        self.peak_materialized_bytes = 0
        self.maximum_simultaneous_materializations = 0

    @property
    def persistent_bytes(self) -> int:
        return sum(a.nbytes for a in (self.parameters, self.gradients, self.first_moment, self.second_moment))

    def materialize(self, name: str) -> np.ndarray:
        if self._active_name is not None:
            raise RuntimeError(f"release tensor {self._active_name!r} before materializing another tensor")
        partition = self.partitions[name]
        pieces = self.group.all_gather(self.parameters[partition.local_start:partition.local_stop])
        for rank, piece in enumerate(pieces):
            lo, hi = shard_bounds(partition.total_size, rank, self.group.size)
            if piece.dtype != np.float32 or piece.shape != (hi - lo,):
                raise ValueError("materialized shard metadata disagrees")
        full = np.concatenate(pieces).reshape(partition.shape)
        self._active_name, self._active_value = name, full
        self.materialization_count += 1
        self.peak_materialized_bytes = max(self.peak_materialized_bytes, full.nbytes)
        self.maximum_simultaneous_materializations = 1
        return full

    def release(self, name: str) -> None:
        if name != self._active_name:
            raise RuntimeError("release must match the currently materialized tensor")
        self._active_name, self._active_value = None, None

    def accumulate_gradient(self, name: str, full_gradient: np.ndarray) -> None:
        partition = self.partitions[name]
        gradient = np.asarray(full_gradient, dtype=np.float32)
        if gradient.shape != partition.shape:
            raise ValueError("gradient shape differs from parameter shape")
        shard = self.group.reduce_scatter(gradient.reshape(-1), reduction=self.gradient_reduction)
        if shard.size != partition.local_stop - partition.local_start:
            raise ValueError("reduced gradient shard has wrong size")
        self.gradients[partition.local_start:partition.local_stop] += shard
        self._accumulations[name] += 1

    def zero_grad(self) -> None:
        self.gradients.fill(0)
        self._accumulations = {name: 0 for name in self.partitions}

    def step(self, max_grad_norm: float | None = None, gradient_scale: float = 1.0) -> dict:
        if self._active_name is not None:
            raise RuntimeError("release materialized tensors before updating parameter shards")
        if not all(self._accumulations.values()):
            raise RuntimeError("every parameter needs a gradient contribution before step")
        if gradient_scale <= 0 or not np.isfinite(gradient_scale):
            raise ValueError("gradient_scale must be positive and finite")
        if max_grad_norm is not None and (max_grad_norm <= 0 or not np.isfinite(max_grad_norm)):
            raise ValueError("max_grad_norm must be positive and finite")
        gradient = self.gradients / np.float32(gradient_scale)
        finite = np.isfinite(gradient).all()
        local_norm2 = float(np.dot(gradient.astype(np.float64), gradient.astype(np.float64))) if finite else 0.0
        statistics = self.group.all_reduce(np.array([int(not finite), local_norm2], dtype=np.float64))
        if statistics[0] or not np.isfinite(statistics[1]):
            self.zero_grad()
            return {"updated": False, "step": self.step_number, "grad_norm": float("inf"), "clip_scale": 0.0}
        norm = float(np.sqrt(statistics[1]))
        clip_scale = min(1.0, max_grad_norm / (norm + 1e-12)) if max_grad_norm is not None else 1.0
        gradient *= np.float32(clip_scale)
        next_step = self.step_number + 1
        beta1, beta2 = self.betas
        # m_t = beta1 m_(t-1) + (1-beta1) g_t.
        next_m = beta1 * self.first_moment + (1.0 - beta1) * gradient
        # v_t = beta2 v_(t-1) + (1-beta2) g_t^2.
        next_v = beta2 * self.second_moment + (1.0 - beta2) * gradient * gradient
        # Bias correction uses the count of accepted updates, not microbatches.
        corrected_m = next_m / (1.0 - beta1 ** next_step)
        corrected_v = next_v / (1.0 - beta2 ** next_step)
        # Decoupled weight decay is outside the Adam preconditioner.
        candidate = self.parameters * (1.0 - self.learning_rate * self.weight_decay)
        candidate -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + self.epsilon)
        failed = not (np.isfinite(candidate).all() and np.isfinite(next_m).all() and np.isfinite(next_v).all())
        if self.group.all_reduce(np.array([int(failed)], dtype=np.int64))[0]:
            self.zero_grad()
            return {"updated": False, "step": self.step_number, "grad_norm": norm, "clip_scale": clip_scale}
        self.parameters[:] = candidate
        self.first_moment[:] = next_m
        self.second_moment[:] = next_v
        self.step_number = next_step
        self._mutation_generation += 1
        self.zero_grad()
        return {"updated": True, "step": self.step_number, "grad_norm": norm, "clip_scale": clip_scale}

    def state_dict(self) -> dict:
        return {"rank": self.group.group_rank, "world_size": self.group.size,
                "names": list(self.partitions), "shapes": [list(p.shape) for p in self.partitions.values()],
                "step": self.step_number, "learning_rate": self.learning_rate,
                "betas": list(self.betas), "epsilon": self.epsilon, "weight_decay": self.weight_decay,
                "gradient_reduction": self.gradient_reduction,
                "parameters": self.parameters.copy(), "gradients": self.gradients.copy(),
                "first_moment": self.first_moment.copy(), "second_moment": self.second_moment.copy(),
                "accumulations": dict(self._accumulations)}

    def load_state_dict(self, state: dict) -> None:
        if self._active_name is not None:
            raise RuntimeError("release materialized tensors before restoring state")
        if (state["rank"] != self.group.group_rank or state["world_size"] != self.group.size
                or state["names"] != list(self.partitions)
                or state["shapes"] != [list(p.shape) for p in self.partitions.values()]):
            raise ValueError("checkpoint partition layout does not match current group")
        for key in ("parameters", "gradients", "first_moment", "second_moment"):
            array = np.asarray(state[key])
            if array.shape != self.parameters.shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"invalid checkpoint array {key}")
            if key == "second_moment" and np.any(array < 0):
                raise ValueError("second moments cannot be negative")
        if type(state["step"]) is not int or state["step"] < 0:
            raise ValueError("invalid optimizer step")
        if set(state["accumulations"]) != set(self.partitions) or any(type(n) is not int or n < 0 for n in state["accumulations"].values()):
            raise ValueError("invalid gradient accumulation counters")
        if (state["learning_rate"] != self.learning_rate or tuple(state["betas"]) != self.betas
                or state["epsilon"] != self.epsilon or state["weight_decay"] != self.weight_decay
                or state["gradient_reduction"] != self.gradient_reduction):
            raise ValueError("checkpoint optimizer hyperparameters disagree")
        for key in ("parameters", "gradients", "first_moment", "second_moment"):
            getattr(self, key)[:] = state[key]
        self.step_number = state["step"]
        self._accumulations = dict(state["accumulations"])
        self._mutation_generation += 1
