"""Real MLA Transformer data-parallel training with NumPy autograd and raw TCP.

The default is two local spawned workers, three AdamW updates, and one BLAS
thread per worker. Every rank owns a full model and Adam state. This entry point
implements ordinary data parallelism; the independent 3D MLP and ZeRO Stage 3
primitives are not silently presented as Transformer pipeline/tensor sharding.
The tiny byte-level C++ text fixture verifies mechanics, not teaching competence.
"""
from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from distributed_demo import run_spawned
from rawllm.distributed import CollectiveServer, ProcessGroup
from rawllm.model import Config, Transformer
from rawllm.optim import AdamW, save_checkpoint
from rawllm.tensor import cross_entropy


_CORPUS = (
    "#include <vector>\n#include <memory>\n"
    "int sum(const std::vector<int>& values) { int total = 0; "
    "for (const int value : values) total += value; return total; }\n"
    "// RAII binds resource lifetime to an object's lifetime.\n"
    "// A compiler parses source, checks types, builds an IR, and emits code.\n"
    "template<class T> T square(T value) { return value * value; }\n"
    "struct Position { float x; float y; };\n"
).encode("utf8")


def configuration() -> Config:
    return Config(vocab_size=256, dim=16, layers=2, heads=2, q_rank=8, kv_rank=6,
                  content_dim=4, rope_dim=4, value_dim=4, hidden_dim=32,
                  max_seq_len=16)


def training_batch(step: int, world_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic global batch; ranks consume disjoint row slices."""
    sequence_length, rows_per_rank = 8, 2
    starts = np.random.default_rng(1234 + step).choice(len(_CORPUS) - sequence_length,
                                                     size=world_size * rows_per_rank, replace=False)
    data = np.frombuffer(_CORPUS, dtype=np.uint8).astype(np.int64)
    sequences = np.stack([data[start:start + sequence_length + 1] for start in starts])
    return sequences[:, :-1], sequences[:, 1:]


def _flatten(parameters: dict, attribute: str) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(parameter, attribute)).reshape(-1)
                           for parameter in parameters.values()])


def _train_rank(rank: int, world_size: int, host: str, port: int, token: str,
                steps: int, checkpoint: str | None) -> dict:
    config = configuration()
    # FP64 arithmetic gives a strict reduction/reference comparison. AdamW
    # explicitly uses FP32 master weights and moment storage in all workers.
    model = Transformer(config, seed=321, dtype=np.float64)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    trace = []
    with ProcessGroup(rank, world_size, host, port, token, timeout=10) as group:
        for step in range(steps):
            global_inputs, global_targets = training_batch(step, world_size)
            row_slice = slice(rank * 2, (rank + 1) * 2)
            optimizer.zero_grad()
            local_loss = cross_entropy(model(global_inputs[row_slice]), global_targets[row_slice])
            local_loss.backward()
            # All local batches contain exactly 16 unmasked target tokens.
            # Mean-of-rank gradients therefore equals the global token mean.
            for name, parameter in model.parameters().items():
                if parameter.grad is None:
                    raise RuntimeError(f"missing Transformer gradient for {name}")
                parameter.grad = group.all_reduce(parameter.grad, reduction="mean")
            mean_loss = group.all_reduce(np.array([local_loss.item()], dtype=np.float64), "mean")[0]
            gradient = _flatten(model.parameters(), "grad")
            if not np.isfinite(gradient).all():
                raise FloatingPointError("distributed Transformer gradient is non-finite")
            optimizer.step()
            trace.append({"loss": float(mean_loss), "gradient": gradient,
                          "weights": _flatten(model.parameters(), "data")})
        if rank == 0 and checkpoint is not None:
            save_checkpoint(checkpoint, model.parameters(), optimizer, np.random.default_rng(1234),
                            {"steps": steps, "world_size": world_size, "config": asdict(config),
                             "objective": "byte-level next-token cross entropy on a tiny C++ fixture",
                             "distributed_mode": "replicated data parallel"})
        group.barrier()
    return {"rank": rank, "trace": trace, "examples_per_step": list(range(rank * 2, (rank + 1) * 2))}


def _single_process_reference(world_size: int, steps: int) -> list[dict]:
    model = Transformer(configuration(), seed=321, dtype=np.float64)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    trace = []
    for step in range(steps):
        inputs, targets = training_batch(step, world_size)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient = _flatten(model.parameters(), "grad")
        optimizer.step()
        trace.append({"loss": loss.item(), "gradient": gradient,
                      "weights": _flatten(model.parameters(), "data")})
    return trace


def run_training(world_size: int = 2, steps: int = 3, checkpoint: str | None = None) -> dict:
    if not 1 <= world_size <= 4 or not 1 <= steps <= 100:
        raise ValueError("local demonstration supports 1..4 ranks and 1..100 updates")
    started = time.monotonic()
    with CollectiveServer(world_size, timeout=10) as server:
        results = run_spawned(_train_rank, world_size,
                              (world_size, server.host, server.port, server.token, steps, checkpoint),
                              timeout=max(25, steps * 2))
    reference = _single_process_reference(world_size, steps)
    maximum_gradient_error = maximum_weight_error = maximum_loss_error = maximum_replica_error = 0.0
    losses = []
    for step, expected in enumerate(reference):
        observed = results[0]["trace"][step]
        maximum_loss_error = max(maximum_loss_error, abs(observed["loss"] - expected["loss"]))
        maximum_gradient_error = max(maximum_gradient_error, float(np.max(np.abs(observed["gradient"] - expected["gradient"]))))
        maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(observed["weights"] - expected["weights"]))))
        for result in results[1:]:
            replica = result["trace"][step]
            maximum_replica_error = max(maximum_replica_error,
                                        float(np.max(np.abs(replica["weights"] - observed["weights"]))))
        losses.append(observed["loss"])
    if maximum_gradient_error > 2e-6 or maximum_weight_error > 2e-6 or maximum_loss_error > 2e-6 or maximum_replica_error:
        raise AssertionError("Transformer data-parallel step differs from combined-batch reference")
    return {"mode": "replicated Transformer data parallelism", "processes": world_size,
            "parameters": configuration().parameter_count, "updates": steps,
            "examples_per_global_batch": world_size * 2, "tokens_per_global_batch": world_size * 16,
            "losses": losses, "maximum_loss_absolute_error": maximum_loss_error,
            "maximum_gradient_absolute_error": maximum_gradient_error,
            "maximum_weight_absolute_error": maximum_weight_error,
            "maximum_replica_absolute_error": maximum_replica_error,
            "reference": "same initialized MLA/RoPE/SwiGLU Transformer on the combined global batch",
            "arithmetic": "FP64 model/autograd; FP32 AdamW master weights and moments",
            "transport": "authenticated coordinator-routed raw TCP on loopback",
            "elapsed_seconds": time.monotonic() - started, "verified": True,
            "checkpoint": checkpoint,
            "scope": "Ordinary DP with full model and optimizer replicas; separate 3D and ZeRO primitives are not integrated here."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    options = parser.parse_args()
    report = run_training(options.world_size, options.steps,
                          str(options.checkpoint.resolve()) if options.checkpoint else None)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
