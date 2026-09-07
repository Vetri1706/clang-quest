"""Execute a tiny 2 DP x 2 PP x 2 TP training step over authenticated TCP.

Run: python distributed_demo.py
Eight spawned CPU processes implement four real MLP matrix layers. Tensor ranks
own column shards of odd layers and row shards of even layers. Pipeline ranks
exchange activations and their gradients. Data ranks consume disjoint examples
and average parameter gradients. The schedule is sequential microbatch
forward/backward, not a throughput-optimized 1F1B schedule. This demonstration
does not automatically partition an arbitrary Transformer or emulate GPU speed.
"""
from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback

import numpy as np

from rawllm.distributed import CollectiveServer, DistributedError, ParallelTopology, ProcessGroup


def run_spawned(target, world_size: int, arguments: tuple, timeout: float = 25.0) -> list:
    """Run independent ranks and tear down all peers on the first worker error."""
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [context.Process(target=_worker_guard, args=(target, rank, arguments, result_queue),
                                 name=f"rawllm-rank-{rank}") for rank in range(world_size)]
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.start()
        results = {}
        while len(results) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DistributedError("spawned job exceeded its wall-clock deadline")
            try:
                rank, ok, payload = result_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                failed = [(p.name, p.exitcode) for p in processes if p.exitcode not in (None, 0)]
                if failed:
                    raise DistributedError(f"worker process failed: {failed}")
                continue
            if not ok:
                raise DistributedError(f"rank {rank} failed:\n{payload}")
            if rank in results:
                raise DistributedError("duplicate worker completion")
            results[rank] = payload
        for process in processes:
            process.join(timeout=max(0.0, min(1.0, deadline - time.monotonic())))
            if process.is_alive() or process.exitcode != 0:
                raise DistributedError(f"worker {process.name} did not shut down cleanly")
        return [results[rank] for rank in range(world_size)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
        result_queue.close()
        result_queue.join_thread()


def _worker_guard(target, rank: int, arguments: tuple, result_queue) -> None:
    try:
        result_queue.put((rank, True, target(rank, *arguments)))
    except BaseException:
        result_queue.put((rank, False, traceback.format_exc()))


def problem() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    weights = [rng.normal(0, 0.2, size=shape).astype(np.float64)
               for shape in ((6, 8), (8, 6), (6, 8), (8, 3))]
    inputs = rng.normal(size=(12, 6)).astype(np.float64)
    targets = rng.normal(size=(12, 3)).astype(np.float64)
    return weights, inputs, targets


def reference() -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    weights, inputs, targets = problem()
    w1, w2, w3, w4 = weights
    h1 = np.tanh(inputs @ w1)
    h2 = np.tanh(h1 @ w2)
    h3 = np.tanh(h2 @ w3)
    output = h3 @ w4
    difference = output - targets
    loss = float(0.5 * np.mean(difference * difference))
    gradient = difference / difference.size
    dw4 = h3.T @ gradient
    gradient = (gradient @ w4.T) * (1.0 - h3 * h3)
    dw3 = h2.T @ gradient
    gradient = (gradient @ w3.T) * (1.0 - h2 * h2)
    dw2 = h1.T @ gradient
    gradient = (gradient @ w2.T) * (1.0 - h1 * h1)
    dw1 = inputs.T @ gradient
    return loss, [dw1, dw2, dw3, dw4], weights


def _training_rank(rank: int, host: str, port: int, token: str) -> dict:
    topology = ParallelTopology()
    dp, pp, tp = topology.coordinates(rank)
    groups = topology.groups(rank)
    weights, inputs, targets = problem()
    microbatches = 2
    data_slice = slice(dp * 6, (dp + 1) * 6)
    local_inputs, local_targets = inputs[data_slice], targets[data_slice]
    feature_slice = slice(tp * 4, (tp + 1) * 4)
    column = weights[2 * pp][:, feature_slice].copy()
    row = weights[2 * pp + 1][feature_slice, :].copy()
    del weights, inputs, targets
    dcolumn, drow = np.zeros_like(column), np.zeros_like(row)
    total_loss = 0.0
    with ProcessGroup(rank, topology.world_size, host, port, token, timeout=12) as world:
        tensor_group, data_group = world.subgroup(groups["tensor"]), world.subgroup(groups["data"])
        for batch in range(microbatches):
            sample_slice = slice(batch * 3, (batch + 1) * 3)
            if pp == 0:
                stage_input = local_inputs[sample_slice]
                hidden = np.tanh(stage_input @ column)
                output = np.tanh(tensor_group.all_reduce(hidden @ row))
                peer = topology.rank(dp, 1, tp)
                world.send(output, peer, tag=f"activation-{batch}")
                gradient = world.recv(peer, tag=f"gradient-{batch}")
                gradient *= 1.0 - output * output
            else:
                peer = topology.rank(dp, 0, tp)
                stage_input = world.recv(peer, tag=f"activation-{batch}")
                hidden = np.tanh(stage_input @ column)
                output = tensor_group.all_reduce(hidden @ row)
                difference = output - local_targets[sample_slice]
                total_loss += float(0.5 * np.mean(difference * difference))
                gradient = difference / difference.size
            drow += (hidden.T @ gradient) / microbatches
            hidden_gradient = (gradient @ row.T) * (1.0 - hidden * hidden)
            dcolumn += (stage_input.T @ hidden_gradient) / microbatches
            # Column-parallel backward reduces the partial input gradients.
            input_gradient = tensor_group.all_reduce(hidden_gradient @ column.T)
            if pp == 1:
                world.send(input_gradient, peer, tag=f"gradient-{batch}")
        dcolumn = data_group.all_reduce(dcolumn, reduction="mean")
        drow = data_group.all_reduce(drow, reduction="mean")
        learning_rate = 0.03
        column -= learning_rate * dcolumn
        row -= learning_rate * drow
        # Every rank joins the same final rendezvous before orderly close.
        world.barrier()
    return {"rank": rank, "coordinate": [dp, pp, tp], "loss": total_loss / microbatches if pp else None,
            "column_gradient": dcolumn, "row_gradient": drow, "column_weight": column, "row_weight": row,
            "examples": list(range(dp * 6, (dp + 1) * 6))}


def run_demo() -> dict:
    start = time.monotonic()
    topology = ParallelTopology()
    with CollectiveServer(topology.world_size, timeout=12) as server:
        results = run_spawned(_training_rank, topology.world_size, (server.host, server.port, server.token), timeout=28)
    reference_loss, reference_gradients, reference_weights = reference()
    reconstructed_gradients, reconstructed_weights = [], []
    for pp in range(topology.pipeline):
        stage = [results[topology.rank(0, pp, tp)] for tp in range(topology.tensor)]
        reconstructed_gradients.extend([np.concatenate([r["column_gradient"] for r in stage], axis=1),
                                        np.concatenate([r["row_gradient"] for r in stage], axis=0)])
        reconstructed_weights.extend([np.concatenate([r["column_weight"] for r in stage], axis=1),
                                      np.concatenate([r["row_weight"] for r in stage], axis=0)])
    measured_loss = float(np.mean([r["loss"] for r in results if r["loss"] is not None]))
    gradient_error = max(float(np.max(np.abs(a - b))) for a, b in zip(reconstructed_gradients, reference_gradients))
    weight_error = max(float(np.max(np.abs(a - (w - 0.03 * g))))
                       for a, w, g in zip(reconstructed_weights, reference_weights, reference_gradients))
    replica_error = 0.0
    for pp in range(2):
        for tp in range(2):
            first, second = results[topology.rank(0, pp, tp)], results[topology.rank(1, pp, tp)]
            for key in ("column_gradient", "row_gradient", "column_weight", "row_weight"):
                replica_error = max(replica_error, float(np.max(np.abs(first[key] - second[key]))))
    loss_error = abs(measured_loss - reference_loss)
    if max(gradient_error, weight_error, loss_error, replica_error) > 1e-11:
        raise AssertionError("distributed training diverged from the single-process analytic reference")
    return {"topology": {"data": 2, "pipeline": 2, "tensor": 2, "processes": 8},
            "transport": "authenticated coordinator-routed TCP over loopback",
            "model": "four-layer float64 tanh MLP, 6-8-6-8-3 dimensions",
            "examples": 12, "microbatches_per_data_rank": 2,
            "reference_loss": reference_loss, "distributed_loss": measured_loss,
            "maximum_gradient_absolute_error": gradient_error,
            "maximum_weight_absolute_error": weight_error,
            "maximum_data_replica_absolute_error": replica_error,
            "elapsed_seconds": time.monotonic() - start,
            "verified": True,
            "scope": "Executable 3D primitives on a tiny MLP; arbitrary Transformer partitioning and GPU transport are not implemented."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write measured JSON results to this path")
    options = parser.parse_args()
    report = run_demo()
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
