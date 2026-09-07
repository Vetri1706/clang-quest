"""Two-rank automatic ZeRO-3 Transformer training and dense-reference audit.

Each worker trains through operation-level weight rematerialization and retains
only parameter/gradient/Adam shards between operators. Small copied shard traces
are returned for this numerical audit; they are verification overhead, not part
of the model's persistent-memory figure. The independent dense reference runs
in the parent after all workers have exited.
"""
from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from distributed_demo import run_spawned
from distributed_train import configuration as small_config, training_batch
from rawllm.distributed import CollectiveServer, ProcessGroup
from rawllm.model import Transformer
from rawllm.optim import AdamW, clip_grad_norm
from rawllm.tensor import cross_entropy
from rawllm.zero import shard_bounds
from rawllm.zero_model import ZeroTransformer


def _worker(rank: int, world_size: int, host: str, port: int, token: str,
            steps: int, tied: bool, checkpoint: str | None, resume: str | None) -> dict:
    config = replace(small_config(), tie_embeddings=tied)
    with ProcessGroup(rank, world_size, host, port, token, timeout=12) as group:
        model = ZeroTransformer(config, group, seed=321, learning_rate=3e-4)
        rng = np.random.default_rng(100 + rank)
        starting_step = 0
        if resume is not None:
            from rawllm.sharded_checkpoint import load_sharded_checkpoint
            counters = load_sharded_checkpoint(resume, model.optimizer, rng)
            starting_step = counters.get("steps")
            if (type(starting_step) is not int or not 0 <= starting_step <= 100
                    or counters.get("rank") != rank or counters.get("tie_embeddings") is not tied
                    or starting_step != model.optimizer.step_number):
                raise ValueError("resume counters disagree with this bounded validation run")
        trace = []
        for step in range(starting_step, starting_step + steps):
            inputs, targets = training_batch(step, world_size)
            model.zero_grad()
            loss_sum = 0.0
            for micro in range(2):
                row = slice(rank * 2 + micro, rank * 2 + micro + 1)
                loss = cross_entropy(model(inputs[row]), targets[row])
                loss_sum += loss.item() / 2
                (loss / 2).backward()
                del loss
            mean_loss = group.all_reduce(np.array([loss_sum], dtype=np.float64), reduction="mean")[0]
            gradients = model.optimizer.gradients.copy()
            accumulations = dict(model.optimizer._accumulations)
            report = model.step(max_grad_norm=0.5)
            if not report["updated"]:
                raise FloatingPointError("finite reference training unexpectedly skipped an update")
            trace.append({"loss": float(mean_loss), "gradients": gradients,
                          "weights": model.optimizer.parameters.copy(),
                          "first_moment": model.optimizer.first_moment.copy(),
                          "second_moment": model.optimizer.second_moment.copy(),
                          "accumulations": accumulations, "grad_norm": report["grad_norm"]})
            if model.optimizer._active_name is not None:
                raise AssertionError("weight materialization survived a training update")
        if checkpoint is not None:
            from rawllm.sharded_checkpoint import save_sharded_checkpoint
            save_sharded_checkpoint(checkpoint, model.optimizer, rng,
                                    {"steps": starting_step + steps, "rank": rank, "tie_embeddings": tied,
                                     "mode": "automatic operation-rematerialized ZeRO-3"})
        group.barrier()
        memory = model.memory_report()
        memory["verification_trace_bytes"] = sum(value.nbytes for row in trace for value in row.values() if isinstance(value, np.ndarray))
        return {"rank": rank, "trace": trace, "memory": memory, "starting_step": starting_step}


def _reference(world_size: int, steps: int, tied: bool) -> list[dict]:
    model = Transformer(replace(small_config(), tie_embeddings=tied), seed=321, dtype=np.float32)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    trace = []
    for step in range(steps):
        inputs, targets = training_batch(step, world_size)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradients = {name: parameter.grad.copy() for name, parameter in model.parameters().items()}
        norm = clip_grad_norm(model.parameters(), max_norm=0.5)
        optimizer.step()
        trace.append({"loss": loss.item(), "gradients": gradients,
                      "weights": {name: parameter.data.copy() for name, parameter in model.parameters().items()},
                      "first_moment": {name: value.copy() for name, value in optimizer.m.items()},
                      "second_moment": {name: value.copy() for name, value in optimizer.v.items()},
                      "grad_norm": norm})
    return trace


def _expected_shard(values: dict[str, np.ndarray], rank: int, world_size: int) -> np.ndarray:
    pieces = []
    for value in values.values():
        start, stop = shard_bounds(value.size, rank, world_size)
        pieces.append(value.reshape(-1)[start:stop])
    return np.concatenate(pieces)


def run_training(world_size: int = 2, steps: int = 3, tied: bool = True,
                 checkpoint: str | None = None, resume: str | None = None) -> dict:
    if not 1 <= world_size <= 4 or not 1 <= steps <= 10:
        raise ValueError("local ZeRO validation supports 1..4 ranks and 1..10 updates")
    started = time.monotonic()
    with CollectiveServer(world_size, timeout=12) as server:
        results = run_spawned(_worker, world_size,
                              (world_size, server.host, server.port, server.token, steps, tied, checkpoint, resume),
                              timeout=28)
    starting_step = results[0]["starting_step"]
    if any(result["starting_step"] != starting_step for result in results):
        raise AssertionError("ranks resumed different training steps")
    reference = _reference(world_size, starting_step + steps, tied)[starting_step:]
    errors = {"loss": 0.0, "gradients": 0.0, "weights": 0.0, "first_moment": 0.0,
              "second_moment": 0.0, "grad_norm": 0.0}
    for rank, result in enumerate(results):
        for actual, expected in zip(result["trace"], reference):
            for field in ("gradients", "weights", "first_moment", "second_moment"):
                desired = _expected_shard(expected[field], rank, world_size)
                errors[field] = max(errors[field], float(np.max(np.abs(actual[field] - desired))))
            for field in ("loss", "grad_norm"):
                errors[field] = max(errors[field], abs(actual[field] - expected[field]))
            expected_embedding_contributions = 4 if tied else 2
            if actual["accumulations"]["embedding"] != expected_embedding_contributions:
                raise AssertionError("tied embedding projection and lookup did not both accumulate")
        memory = result["memory"]
        if memory["maximum_simultaneous_materialized_tensors"] != 1 or memory["active_materialization"] is not None:
            raise AssertionError("operator-level materialization lifetime exceeded one tensor")
        if memory["persistent_parameter_gradient_adam_bytes"] != memory["locally_owned_parameters"] * 16:
            raise AssertionError("persistent optimizer state is not partitioned")
    if any(error > 3e-5 for error in errors.values()):
        raise AssertionError(f"automatic ZeRO Transformer diverged from dense reference: {errors}")
    return {"mode": "automatic operation-rematerialized ZeRO-3 Transformer", "world_size": world_size,
            "parameters": replace(small_config(), tie_embeddings=tied).parameter_count,
            "tie_embeddings": tied, "updates": steps, "starting_step": starting_step,
            "completed_steps": starting_step + steps, "microbatches_per_rank": 2,
            "losses": [row["loss"] for row in results[0]["trace"]],
            "maximum_absolute_errors": errors,
            "memory_by_rank": [result["memory"] for result in results],
            "reference": "regular FP32 Transformer and AdamW on combined global batches with global norm clipping",
            "verified": True, "elapsed_seconds": time.monotonic() - started,
            "sharded_checkpoint": checkpoint, "resumed_from": resume,
            "scope": "automatic ZeRO-3 for the Config MLA architecture across data ranks; activation checkpointing and combination with pipeline/tensor parallelism are not implemented"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--untied", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path, help="restore a matching sharded checkpoint and continue its update count")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = run_training(options.world_size, options.steps, not options.untied,
                          str(options.checkpoint.resolve()) if options.checkpoint else None,
                          str(options.resume.resolve()) if options.resume else None)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
