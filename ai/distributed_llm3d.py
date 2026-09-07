"""Executable 2 DP x 2 PP x 2 TP training of the actual MLA Transformer.

Attention heads and SwiGLU hidden features are genuinely sharded; pipeline ranks
own distinct layers and exchange detached activation/gradient tensors over raw
TCP. Shared latent projections are replicated within each tensor group and
their parameter gradients are summed explicitly. No automatic ZeRO scheduling,
RDMA, communication/computation overlap, or arbitrary partition planner is
claimed. The fixed tiny configuration is designed for a local Mac verification.
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
from distributed_train import training_batch
from rawllm.distributed import CollectiveServer, ParallelTopology, ProcessGroup
from rawllm.model import Config, Transformer
from rawllm.optim import AdamW
from rawllm.tensor import Tensor, Parameter, attention, concatenate, cross_entropy, embedding, rms_norm, rope


_COLUMN_SHARDS = {"q_up", "k_up", "v_up", "gate", "up"}
_ROW_SHARDS = {"o", "down"}
_SUM_REPLICATED_GRADIENTS = {"q_down", "q_norm", "kv_down", "kv_norm", "k_rope"}


def configuration() -> Config:
    return Config(vocab_size=256, dim=16, layers=2, heads=2, q_rank=8, kv_rank=6,
                  content_dim=4, rope_dim=4, value_dim=4, hidden_dim=32,
                  max_seq_len=16, tie_embeddings=False)


def copy_to_tensor_parallel(value: Tensor, group: ProcessGroup) -> Tensor:
    """Replicated forward value; sum downstream partial input gradients."""
    return Tensor._from_op(value.data, (value,), lambda gradient: (group.all_reduce(gradient),))


def reduce_from_tensor_parallel(value: Tensor, group: ProcessGroup) -> Tensor:
    """Sum partial forward outputs; distribute the same output cotangent."""
    return Tensor._from_op(group.all_reduce(value.data), (value,), lambda gradient: (gradient,))


def parameter_stage(name: str, topology: ParallelTopology) -> int:
    if name == "embedding":
        return 0
    if name in {"final_norm", "head"}:
        return topology.pipeline - 1
    if not name.startswith("blocks."):
        raise ValueError(f"unrecognized parameter name {name}")
    layers_per_stage = configuration().layers // topology.pipeline
    return int(name.split(".")[1]) // layers_per_stage


def partition_axis(name: str) -> int | None:
    suffix = name.rsplit(".", 1)[-1]
    if name.startswith("blocks.") and suffix in _COLUMN_SHARDS:
        return 1
    if name.startswith("blocks.") and suffix in _ROW_SHARDS:
        return 0
    return None


class TransformerStage:
    """One pipeline stage with locally owned attention and FFN weight shards.

    The temporary full initialization is freed after owned slices are copied.
    The persistent stage contains no parameters from other pipeline stages.
    Attention latent projections, normalization scales, embedding, and output
    head remain replicated wherever their operation needs a full input vector.
    """

    def __init__(self, pipeline_rank: int, tensor_rank: int, tensor_group: ProcessGroup,
                 topology: ParallelTopology):
        self.config, self.topology = configuration(), topology
        self.pipeline_rank, self.tensor_rank = pipeline_rank, tensor_rank
        self.tensor_group = tensor_group
        if self.config.layers % topology.pipeline or self.config.heads % topology.tensor or self.config.hidden_dim % topology.tensor:
            raise ValueError("layer, attention-head and FFN dimensions must divide the fixed topology")
        full = Transformer(self.config, seed=321, dtype=np.float64)
        self.params = {}
        for name, parameter in full.parameters().items():
            if parameter_stage(name, topology) != pipeline_rank:
                continue
            axis = partition_axis(name)
            value = parameter.data if axis is None else np.array_split(parameter.data, topology.tensor, axis=axis)[tensor_rank]
            self.params[name] = Parameter(value.copy(), name=name)
        del full
        layers_per_stage = self.config.layers // topology.pipeline
        self.layers = range(pipeline_rank * layers_per_stage, (pipeline_rank + 1) * layers_per_stage)

    def block(self, x: Tensor, index: int) -> Tensor:
        c = self.config
        prefix = f"blocks.{index}."
        weight = lambda key: self.params[prefix + key]
        batch, length, _ = x.shape
        local_heads = c.heads // self.topology.tensor
        positions = np.arange(length)
        # RMSNorm is evaluated once per replica, outside the partial-head map;
        # its scale gradient already receives the sum over all local heads.
        u = copy_to_tensor_parallel(rms_norm(x, weight("attn_norm"), c.norm_eps), self.tensor_group)
        query_latent = rms_norm(u @ weight("q_down"), weight("q_norm"), c.norm_eps)
        query = (query_latent @ weight("q_up")).reshape(batch, length, local_heads,
                                                        c.content_dim + c.rope_dim).transpose(0, 2, 1, 3)
        query_content = query[..., :c.content_dim]
        query_rotary = rope(query[..., c.content_dim:], positions, c.rope_base)
        latent = rms_norm(u @ weight("kv_down"), weight("kv_norm"), c.norm_eps)
        key_content = (latent @ weight("k_up")).reshape(batch, length, local_heads, c.content_dim).transpose(0, 2, 1, 3)
        key_rotary = rope((u @ weight("k_rope")).reshape(batch, 1, length, c.rope_dim), positions, c.rope_base)
        key_rotary = key_rotary * np.ones((1, local_heads, 1, 1), dtype=x.dtype)
        values = (latent @ weight("v_up")).reshape(batch, length, local_heads, c.value_dim).transpose(0, 2, 1, 3)
        attended = attention(concatenate((query_content, query_rotary), axis=-1),
                             concatenate((key_content, key_rotary), axis=-1), values,
                             causal=True, block_size=32)
        partial_output = attended.transpose(0, 2, 1, 3).reshape(batch, length, local_heads * c.value_dim) @ weight("o")
        x = x + reduce_from_tensor_parallel(partial_output, self.tensor_group)
        u = copy_to_tensor_parallel(rms_norm(x, weight("ffn_norm"), c.norm_eps), self.tensor_group)
        partial_ffn = ((u @ weight("gate")).silu() * (u @ weight("up"))) @ weight("down")
        return x + reduce_from_tensor_parallel(partial_ffn, self.tensor_group)

    def __call__(self, stage_input: Tensor | np.ndarray) -> Tensor:
        if self.pipeline_rank == 0:
            x = embedding(self.params["embedding"], np.asarray(stage_input))
        else:
            if not isinstance(stage_input, Tensor) or not stage_input.requires_grad:
                raise ValueError("later pipeline stages require a differentiable activation leaf")
            x = stage_input
        for index in self.layers:
            x = self.block(x, index)
        if self.pipeline_rank == self.topology.pipeline - 1:
            x = rms_norm(x, self.params["final_norm"], self.config.norm_eps) @ self.params["head"]
        return x

    def synchronize_gradients(self, data_group: ProcessGroup) -> None:
        for name, parameter in self.params.items():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient for owned parameter {name}")
            # These full latent matrices/scales participate inside the partial
            # head computation; their local gradient omits the other TP heads.
            if name.rsplit(".", 1)[-1] in _SUM_REPLICATED_GRADIENTS:
                parameter.grad = self.tensor_group.all_reduce(parameter.grad)
            parameter.grad = data_group.all_reduce(parameter.grad, reduction="mean")


def _rank_train(rank: int, host: str, port: int, token: str, steps: int) -> dict:
    topology = ParallelTopology(2, 2, 2)
    dp, pp, tp = topology.coordinates(rank)
    with ProcessGroup(rank, topology.world_size, host, port, token, timeout=12) as world:
        groups = topology.groups(rank)
        tensor_group, data_group = world.subgroup(groups["tensor"]), world.subgroup(groups["data"])
        stage = TransformerStage(pp, tp, tensor_group, topology)
        optimizer = AdamW(stage.params, lr=3e-4, weight_decay=0.01)
        trace = []
        for step in range(steps):
            inputs, targets = training_batch(step, topology.data)
            optimizer.zero_grad()
            local_loss = 0.0
            microbatches = 2
            for micro in range(microbatches):
                tag = f"step-{step}-micro-{micro}"
                row = slice(dp * 2 + micro, dp * 2 + micro + 1)
                if pp == 0:
                    stage_output = stage(inputs[row])
                    next_rank = topology.rank(dp, pp + 1, tp)
                    world.send(stage_output.data, next_rank, f"activation-{tag}")
                    output_gradient = world.recv(next_rank, f"gradient-{tag}")
                    stage_output.backward(output_gradient)
                else:
                    previous_rank = topology.rank(dp, pp - 1, tp)
                    stage_input = Tensor(world.recv(previous_rank, f"activation-{tag}"), requires_grad=True)
                    stage_output = stage(stage_input)
                    loss = cross_entropy(stage_output, targets[row])
                    local_loss += loss.item() / microbatches
                    # Mean over equally sized microbatches before the data-rank
                    # mean; the gradient sent to stage zero is already scaled.
                    (loss / microbatches).backward()
                    if stage_input.grad is None:
                        raise RuntimeError("pipeline input leaf did not receive a gradient")
                    world.send(stage_input.grad, previous_rank, f"gradient-{tag}")
                    del stage_input, loss
                del stage_output
            stage.synchronize_gradients(data_group)
            gradient = {name: parameter.grad.copy() for name, parameter in stage.params.items()}
            optimizer.step()
            trace.append({"loss": local_loss if pp == 1 else None, "gradients": gradient,
                          "weights": {name: parameter.data.copy() for name, parameter in stage.params.items()}})
        world.barrier()
        return {"rank": rank, "coordinates": [dp, pp, tp], "trace": trace,
                "owned_parameters": sum(parameter.size for parameter in stage.params.values()),
                "owned_names": list(stage.params)}


def _reference(steps: int) -> list[dict]:
    model = Transformer(configuration(), seed=321, dtype=np.float64)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    trace = []
    for step in range(steps):
        inputs, targets = training_batch(step, world_size=2)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradients = {name: parameter.grad.copy() for name, parameter in model.parameters().items()}
        optimizer.step()
        trace.append({"loss": loss.item(), "gradients": gradients,
                      "weights": {name: parameter.data.copy() for name, parameter in model.parameters().items()}})
    return trace


def _reconstruct(results: list[dict], topology: ParallelTopology, step: int,
                 field: str, data_rank: int) -> dict[str, np.ndarray]:
    complete = {}
    for name in configuration().shapes():
        pp = parameter_stage(name, topology)
        values = [results[topology.rank(data_rank, pp, tp)]["trace"][step][field][name]
                  for tp in range(topology.tensor)]
        axis = partition_axis(name)
        if axis is None:
            if any(not np.array_equal(values[0], value) for value in values[1:]):
                raise AssertionError(f"tensor replicas disagree for {field}/{name}")
            complete[name] = values[0]
        else:
            complete[name] = np.concatenate(values, axis=axis)
    return complete


def run_training(steps: int = 2, checkpoint: str | None = None) -> dict:
    if not 1 <= steps <= 10:
        raise ValueError("the local 3D validation run supports 1..10 updates")
    started = time.monotonic()
    topology = ParallelTopology(2, 2, 2)
    with CollectiveServer(topology.world_size, timeout=12) as server:
        results = run_spawned(_rank_train, topology.world_size,
                              (server.host, server.port, server.token, steps), timeout=28)
    reference = _reference(steps)
    errors = {"loss": 0.0, "gradient": 0.0, "weight": 0.0, "data_replica": 0.0}
    losses = []
    final_weights = {}
    for step, expected in enumerate(reference):
        loss = float(np.mean([result["trace"][step]["loss"] for result in results
                              if result["trace"][step]["loss"] is not None]))
        losses.append(loss)
        errors["loss"] = max(errors["loss"], abs(loss - expected["loss"]))
        for field, error_name in (("gradients", "gradient"), ("weights", "weight")):
            reconstructed = _reconstruct(results, topology, step, field, 0)
            other_data_rank = _reconstruct(results, topology, step, field, 1)
            for name in reconstructed:
                errors[error_name] = max(errors[error_name], float(np.max(np.abs(reconstructed[name] - expected[field][name]))))
                errors["data_replica"] = max(errors["data_replica"], float(np.max(np.abs(reconstructed[name] - other_data_rank[name]))))
            if field == "weights":
                final_weights = reconstructed
    if any(error > 2e-6 for error in errors.values()) or errors["data_replica"]:
        raise AssertionError(f"3D Transformer differs from dense combined-batch reference: {errors}")
    if checkpoint:
        directory = Path(checkpoint)
        directory.mkdir(parents=True, exist_ok=True)
        # This reconstructed export contains model weights only. Worker Adam
        # states are not exported, so this is not an optimizer-resume checkpoint.
        with (directory / "weights.npz").open("wb") as handle:
            np.savez(handle, **final_weights)
        (directory / "config.json").write_text(json.dumps(asdict(configuration()), indent=2) + "\n", encoding="utf8")
    return {"mode": "real MLA Transformer 3D parallel training",
            "topology": {"data": 2, "pipeline": 2, "tensor": 2, "processes": 8},
            "parameters": configuration().parameter_count,
            "locally_owned_parameters_by_rank": [result["owned_parameters"] for result in results],
            "updates": steps, "microbatches_per_data_rank": 2,
            "global_sequences_per_update": 4, "global_target_tokens_per_update": 32,
            "losses": losses, "maximum_loss_absolute_error": errors["loss"],
            "maximum_gradient_absolute_error": errors["gradient"],
            "maximum_weight_absolute_error": errors["weight"],
            "maximum_data_replica_absolute_error": errors["data_replica"],
            "tensor_replicas_exact": True,
            "reference": "full unpartitioned MLA/RoPE/SwiGLU Transformer on combined global batches",
            "arithmetic": "FP64 model/autograd; FP32 AdamW masters and moments",
            "transport": "HMAC-authenticated coordinator-routed raw TCP over loopback",
            "elapsed_seconds": time.monotonic() - started, "verified": True,
            "weight_export": checkpoint,
            "scope": "Fixed tiny untied-embedding Transformer, sequential two-stage schedule, head/FFN tensor shards, replicated latent projections; no automatic ZeRO integration or optimized GPU transport."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path, help="export reconstructed trained weights and config; optimizer state is excluded")
    options = parser.parse_args()
    report = run_training(options.steps, str(options.weights.resolve()) if options.weights else None)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
