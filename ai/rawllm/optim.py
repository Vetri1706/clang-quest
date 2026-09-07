"""FP32 AdamW, emulated reduced precision, loss scaling and safe checkpoints.

Precision emulation rounds storage values; NumPy kernels and optimizer master
weights remain FP32. This explicitly does not claim native BF16 tensor-core speed.
Checkpoint NPZ payloads prohibit pickle and are committed through one atomic JSON
manifest rename after file fsync. Checkpoint directories are trusted local input;
checksums detect corruption, not an attacker who can replace both files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping
import uuid

import numpy as np

from .tensor import Tensor
from .safeio import read_json, read_npz


def _parameters(parameters: Mapping[str, Tensor] | Iterable[Tensor]) -> dict[str, Tensor]:
    result = dict(parameters) if isinstance(parameters, Mapping) else {str(i): p for i, p in enumerate(parameters)}
    if not result or not all(isinstance(name, str) and isinstance(p, Tensor) for name, p in result.items()):
        raise ValueError("Parameters must be a nonempty mapping of names to tensors")
    if len({id(parameter) for parameter in result.values()}) != len(result):
        raise ValueError("Tied parameters must appear only once in the optimizer")
    return result


class AdamW:
    """Decoupled weight decay with FP32 masters and per-parameter update clocks."""

    def __init__(self, parameters: Mapping[str, Tensor] | Iterable[Tensor], lr: float = 3e-4,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("lr and eps must be finite and positive")
        if len(betas) != 2 or not all(np.isfinite(b) and 0 <= b < 1 for b in betas):
            raise ValueError("Adam beta values must be in [0,1)")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        self.parameters = _parameters(parameters)
        if any(not np.issubdtype(p.data.dtype, np.floating) for p in self.parameters.values()):
            raise TypeError("AdamW parameters must have floating-point storage")
        self.lr, self.betas, self.eps, self.weight_decay = float(lr), tuple(betas), float(eps), float(weight_decay)
        self.master = {name: p.data.astype(np.float32, copy=True) for name, p in self.parameters.items()}
        if any(not np.isfinite(value).all() for value in self.master.values()):
            raise ValueError("Optimizer parameters must be finite in FP32 master storage")
        self.m = {name: np.zeros_like(master) for name, master in self.master.items()}
        self.v = {name: np.zeros_like(master) for name, master in self.master.items()}
        self.steps = {name: 0 for name in self.parameters}
        self.step_count = 0

    def zero_grad(self) -> None:
        for parameter in self.parameters.values():
            parameter.grad = None

    def step(self) -> None:
        active: list[tuple[str, Tensor, np.ndarray]] = []
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=np.float32)
            if gradient.shape != parameter.shape:
                raise ValueError(f"Gradient shape mismatch for {name}")
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(f"Non-finite gradient for {name}; no parameters were updated")
            active.append((name, parameter, gradient))
        if not active:
            return
        beta1, beta2 = self.betas
        candidates = {}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for name, parameter, gradient in active:
                step = self.steps[name] + 1
                moment, variance, master = self.m[name].copy(), self.v[name].copy(), self.master[name].copy()
                # m_t = beta1*m_(t-1) + (1-beta1)*g_t
                moment *= beta1
                moment += (1.0 - beta1) * gradient
                # v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
                variance *= beta2
                variance += (1.0 - beta2) * gradient * gradient
                first_moment = moment / (1.0 - beta1 ** step)
                second_moment = variance / (1.0 - beta2 ** step)
                # theta_t = (1-lr*lambda)*theta_(t-1) - lr*mhat/(sqrt(vhat)+eps)
                master *= 1.0 - self.lr * self.weight_decay
                master -= self.lr * first_moment / (np.sqrt(second_moment) + self.eps)
                visible = master.astype(parameter.data.dtype)
                if any(not np.isfinite(array).all() for array in (moment, variance, master, visible)) or np.any(variance < 0):
                    raise FloatingPointError(f"Non-finite candidate update for {name}; no optimizer state was changed")
                candidates[name] = moment, variance, master, visible, step
        # Commit only after every candidate succeeds, including lower precision casts.
        for name, parameter, _ in active:
            moment, variance, master, visible, step = candidates[name]
            self.m[name][...] = moment
            self.v[name][...] = variance
            self.master[name][...] = master
            parameter.data[...] = visible
            self.steps[name] = step
        self.step_count += 1

    def state_dict(self) -> dict:
        return {"lr": self.lr, "betas": list(self.betas), "eps": self.eps,
                "weight_decay": self.weight_decay, "step_count": self.step_count,
                "steps": dict(self.steps), "master": {k: v.copy() for k, v in self.master.items()},
                "m": {k: v.copy() for k, v in self.m.items()}, "v": {k: v.copy() for k, v in self.v.items()}}

    def load_state_dict(self, state: Mapping) -> None:
        required = {"lr", "betas", "eps", "weight_decay", "step_count", "steps", "master", "m", "v"}
        if set(state) != required:
            raise ValueError("Optimizer state fields do not match AdamW schema")
        probe_lr, probe_eps, probe_decay = float(state["lr"]), float(state["eps"]), float(state["weight_decay"])
        probe_betas = tuple(float(b) for b in state["betas"])
        if not np.isfinite(probe_lr) or probe_lr <= 0 or not np.isfinite(probe_eps) or probe_eps <= 0:
            raise ValueError("Invalid optimizer lr/eps in checkpoint")
        if not np.isfinite(probe_decay) or probe_decay < 0 or len(probe_betas) != 2 or not all(0 <= b < 1 for b in probe_betas):
            raise ValueError("Invalid optimizer beta/decay in checkpoint")
        if type(state["step_count"]) is not int or state["step_count"] < 0 or set(state["steps"]) != set(self.parameters):
            raise ValueError("Invalid optimizer step counters")
        restored: dict[str, dict[str, np.ndarray]] = {}
        for group in ("master", "m", "v"):
            if set(state[group]) != set(self.parameters):
                raise ValueError(f"Optimizer {group} parameter names mismatch")
            restored[group] = {}
            for name, parameter in self.parameters.items():
                array = np.asarray(state[group][name])
                if array.shape != parameter.shape or array.dtype != np.float32 or not np.all(np.isfinite(array)):
                    raise ValueError(f"Invalid optimizer {group} state for {name}")
                if group == "v" and np.any(array < 0):
                    raise ValueError("Adam second moments cannot be negative")
                restored[group][name] = array.copy()
        if any(type(value) is not int for value in state["steps"].values()):
            raise ValueError("Per-parameter step counters must be integers")
        steps = dict(state["steps"])
        if any(value < 0 or value > int(state["step_count"]) for value in steps.values()):
            raise ValueError("Invalid per-parameter step counter")
        self.lr, self.eps, self.weight_decay, self.betas = probe_lr, probe_eps, probe_decay, probe_betas
        self.master, self.m, self.v = restored["master"], restored["m"], restored["v"]
        self.steps, self.step_count = steps, int(state["step_count"])


def clip_grad_norm(parameters: Mapping[str, Tensor] | Iterable[Tensor], max_norm: float,
                   epsilon: float = 1e-12) -> float:
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    gradients = [parameter.grad for parameter in _parameters(parameters).values() if parameter.grad is not None]
    norm = float(np.sqrt(sum(float(np.sum(np.square(np.asarray(gradient, dtype=np.float64)))) for gradient in gradients)))
    if not np.isfinite(norm):
        raise FloatingPointError("Cannot clip non-finite gradients")
    scale = min(1.0, max_norm / (norm + epsilon))
    for gradient in gradients:
        gradient *= scale
    return norm


def accumulate_gradients(losses: Iterable[Tensor], parameters: Mapping[str, Tensor] | Iterable[Tensor]) -> int:
    """Backpropagate a mean of scalar microbatch losses without retaining graphs."""
    params = _parameters(parameters)
    for parameter in params.values():
        parameter.grad = None
    count = 0
    for loss in losses:
        if loss.data.size != 1:
            raise ValueError("Gradient accumulation expects scalar microbatch losses")
        loss.backward()
        count += 1
    if count == 0:
        raise ValueError("Gradient accumulation requires a microbatch")
    for parameter in params.values():
        if parameter.grad is not None:
            parameter.grad /= count
    return count


def quantize_array(array: np.ndarray, precision: str = "bf16") -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if precision == "fp32":
        return values.copy()
    if precision == "fp16":
        with np.errstate(over="ignore", invalid="ignore"):
            return values.astype(np.float16).astype(np.float32)
    if precision != "bf16":
        raise ValueError("precision must be fp32, fp16 or bf16")
    bits = values.view(np.uint32)
    # Add half an omitted unit, with retained LSB correcting ties to even.
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isfinite(values), rounded, values).astype(np.float32)


def quantize_ste(tensor: Tensor, precision: str = "bf16") -> Tensor:
    """Round forward values; use the identity straight-through gradient estimator."""
    rounded = quantize_array(tensor.data, precision)
    return Tensor._from_op(rounded, (tensor,), lambda gradient: (gradient,))


class DynamicLossScaler:
    def __init__(self, initial_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000,
                 min_scale: float = 1.0, max_scale: float = 2.0 ** 24):
        if not all(np.isfinite(x) for x in (initial_scale, growth_factor, backoff_factor, min_scale, max_scale)):
            raise ValueError("Loss scaler values must be finite")
        if not 0 < min_scale <= initial_scale <= max_scale or growth_factor <= 1 or not 0 < backoff_factor < 1 or growth_interval < 1:
            raise ValueError("Invalid dynamic loss scaler configuration")
        self.scale_value = float(initial_scale)
        self.growth_factor, self.backoff_factor = float(growth_factor), float(backoff_factor)
        self.growth_interval, self.min_scale, self.max_scale = int(growth_interval), float(min_scale), float(max_scale)
        self.good_steps = 0

    def scale(self, loss: Tensor) -> Tensor:
        return loss * self.scale_value

    def unscale_(self, parameters: Mapping[str, Tensor] | Iterable[Tensor]) -> bool:
        """Divide gradients once and return True when all gradients are finite."""
        found_inf = False
        for parameter in _parameters(parameters).values():
            if parameter.grad is not None:
                parameter.grad /= self.scale_value
                found_inf = found_inf or not bool(np.all(np.isfinite(parameter.grad)))
        return not found_inf

    def update(self, finite: bool) -> None:
        if not finite:
            self.scale_value = max(self.min_scale, self.scale_value * self.backoff_factor)
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps >= self.growth_interval:
                self.scale_value = min(self.max_scale, self.scale_value * self.growth_factor)
                self.good_steps = 0

    def step(self, optimizer: AdamW, max_norm: float | None = None) -> bool:
        finite = self.unscale_(optimizer.parameters)
        if finite:
            if max_norm is not None:
                clip_grad_norm(optimizer.parameters, max_norm)
            optimizer.step()
        self.update(finite)
        optimizer.zero_grad()
        return finite

    def state_dict(self) -> dict:
        return {"scale": self.scale_value, "growth_factor": self.growth_factor,
                "backoff_factor": self.backoff_factor, "growth_interval": self.growth_interval,
                "min_scale": self.min_scale, "max_scale": self.max_scale, "good_steps": self.good_steps}

    def load_state_dict(self, state: Mapping) -> None:
        clone = DynamicLossScaler(initial_scale=state["scale"], growth_factor=state["growth_factor"],
                                  backoff_factor=state["backoff_factor"], growth_interval=state["growth_interval"],
                                  min_scale=state["min_scale"], max_scale=state["max_scale"])
        clone.good_steps = int(state["good_steps"])
        if clone.good_steps < 0 or clone.good_steps >= clone.growth_interval:
            raise ValueError("Invalid loss scaler growth counter")
        self.__dict__.update(clone.__dict__)


def _json_encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.dtype.itemsize > 8 or value.ndim > 16 or not np.isfinite(value).all():
            raise ValueError("Checkpoint JSON arrays must contain finite plain real numeric values")
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_encode(item) for item in value]
    return value


def _json_decode(value):
    if isinstance(value, dict):
        if set(value) == {"__ndarray__", "dtype"}:
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biuf" or dtype.itemsize > 8 or dtype.fields or dtype.subdtype or dtype.hasobject:
                raise ValueError("Invalid checkpoint JSON numerical dtype")
            array = np.asarray(value["__ndarray__"], dtype=dtype)
            if array.ndim > 16 or not np.isfinite(array).all():
                raise ValueError("Invalid checkpoint JSON numerical values")
            return array
        return {key: _json_decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_decode(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(path: str | Path, parameters: Mapping[str, Tensor], optimizer: AdamW,
                    rng: np.random.Generator, counters: Mapping,
                    scaler: DynamicLossScaler | None = None) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    _sync_directory(directory.parent)
    params = _parameters(parameters)
    if set(params) != set(optimizer.parameters) or any(params[name] is not optimizer.parameters[name] for name in params):
        raise ValueError("Checkpoint and optimizer must reference the same named parameter objects")
    arrays: dict[str, np.ndarray] = {}
    names = sorted(params)
    gradients: list[str] = []
    for index, name in enumerate(names):
        parameter = params[name]
        arrays[f"parameter_{index}"] = parameter.data
        if parameter.grad is not None:
            arrays[f"gradient_{index}"] = parameter.grad
            gradients.append(name)
        for group in ("master", "m", "v"):
            arrays[f"{group}_{index}"] = getattr(optimizer, group)[name]
    opt_metadata = {"lr": optimizer.lr, "betas": list(optimizer.betas), "eps": optimizer.eps,
                    "weight_decay": optimizer.weight_decay, "step_count": optimizer.step_count,
                    "steps": dict(optimizer.steps)}
    identifier = uuid.uuid4().hex
    payload = directory / f"weights-{identifier}.npz"
    partial_payload = directory / f".weights-{identifier}.partial"
    metadata = {"format": "rawllm-checkpoint", "version": 1, "parameters": names,
                "gradients": gradients, "optimizer": opt_metadata,
                "rng": _json_encode(rng.bit_generator.state), "counters": _json_encode(dict(counters)),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "payload": payload.name}
    # Validate JSON serialization before creating a new immutable payload.
    json.dumps(metadata, allow_nan=False)
    temporary_manifest: Path | None = None
    try:
        with partial_payload.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial_payload, payload)
        metadata["sha256"] = _sha256(payload)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".manifest-", delete=False) as handle:
            temporary_manifest = Path(handle.name)
            json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, directory / "manifest.json")
        _sync_directory(directory)
    finally:
        if partial_payload.exists():
            partial_payload.unlink()
        if temporary_manifest is not None and temporary_manifest.exists():
            temporary_manifest.unlink()
    return directory / "manifest.json"


def load_checkpoint(path: str | Path, parameters: Mapping[str, Tensor], optimizer: AdamW,
                    rng: np.random.Generator, scaler: DynamicLossScaler | None = None) -> dict:
    directory = Path(path)
    metadata = read_json(directory / "manifest.json")
    if metadata.get("format") != "rawllm-checkpoint" or metadata.get("version") != 1:
        raise ValueError("Unsupported checkpoint format")
    payload_name = metadata["payload"]
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name or not payload_name.endswith(".npz"):
        raise ValueError("Invalid checkpoint payload path")
    payload = directory / payload_name
    params = _parameters(parameters)
    if sorted(params) != metadata["parameters"] or set(params) != set(optimizer.parameters):
        raise ValueError("Checkpoint parameter names mismatch")
    if any(params[name] is not optimizer.parameters[name] for name in params):
        raise ValueError("Checkpoint optimizer does not own the supplied parameters")
    if len(set(metadata["gradients"])) != len(metadata["gradients"]) or not set(metadata["gradients"]).issubset(params):
        raise ValueError("Checkpoint gradient names mismatch")
    values: dict[str, np.ndarray] = {}
    gradients: dict[str, np.ndarray | None] = {}
    state = dict(metadata["optimizer"])
    state.update({group: {} for group in ("master", "m", "v")})
    expected = {}
    for index, name in enumerate(metadata["parameters"]):
        parameter = params[name]
        expected[f"parameter_{index}"] = (parameter.shape, parameter.data.dtype)
        if name in metadata["gradients"]:
            expected[f"gradient_{index}"] = (parameter.shape, (np.float16, np.float32, np.float64))
        for group in ("master", "m", "v"):
            expected[f"{group}_{index}"] = (parameter.shape, np.float32)
    archive = read_npz(payload, expected, sha256=metadata["sha256"])
    expected_keys: set[str] = set()
    for index, name in enumerate(metadata["parameters"]):
        parameter = params[name]
        key = f"parameter_{index}"
        expected_keys.add(key)
        array = archive[key]
        if array.shape != parameter.shape or array.dtype != parameter.data.dtype or not np.all(np.isfinite(array)):
            raise ValueError(f"Checkpoint parameter shape, dtype, or finite-value mismatch: {name}")
        values[name] = array.copy()
        gradients[name] = None
        if name in metadata["gradients"]:
            key = f"gradient_{index}"
            expected_keys.add(key)
            gradient = archive[key]
            if gradient.shape != parameter.shape or not np.issubdtype(gradient.dtype, np.floating) or not np.all(np.isfinite(gradient)):
                raise ValueError(f"Checkpoint gradient invalid: {name}")
            gradients[name] = gradient.copy()
        for group in ("master", "m", "v"):
            key = f"{group}_{index}"
            expected_keys.add(key)
            state[group][name] = archive[key].copy()
    if set(archive) != expected_keys:
        raise ValueError("Checkpoint array inventory does not match manifest")
    rng_state = _json_decode(metadata["rng"])
    if rng_state.get("bit_generator") != type(rng.bit_generator).__name__:
        raise ValueError("Checkpoint uses a different NumPy random bit generator")
    # Validate auxiliary state before mutating any live tensors.
    trial_rng = type(rng.bit_generator)()
    trial_rng.state = rng_state
    restored_scaler = None
    if metadata["scaler"] is not None:
        if scaler is None:
            raise ValueError("Checkpoint contains a loss scaler but caller did not provide one")
        restored_scaler = DynamicLossScaler()
        restored_scaler.load_state_dict(metadata["scaler"])
    elif scaler is not None:
        raise ValueError("Caller provided a loss scaler absent from checkpoint")
    counters = _json_decode(metadata["counters"])
    if not isinstance(counters, dict):
        raise ValueError("Checkpoint counters must be a mapping")
    optimizer.load_state_dict(state)
    for name, parameter in params.items():
        parameter.data[...] = values[name]
        parameter.grad = gradients[name]
    rng.bit_generator.state = rng_state
    if restored_scaler is not None:
        scaler.load_state_dict(restored_scaler.state_dict())
    return counters
