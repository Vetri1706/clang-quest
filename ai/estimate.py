"""Analytical capacity and topology model; allocates no Transformer arrays."""
import argparse
import json
import math
from dataclasses import asdict
from rawllm.model import Config


def estimate(config, dp=1, tp=1, pp=1, batch=1, sequence=2048, weight_bytes=2,
             latency_us=5.0, link_gbps=200.0, page_size=16):
    if (any(type(n) is not int or n < 1 for n in (dp, tp, pp, batch, sequence, weight_bytes, page_size))
            or not math.isfinite(link_gbps) or not math.isfinite(latency_us) or link_gbps <= 0 or latency_us < 0):
        raise ValueError("Positive dimensions/bandwidth and nonnegative latency required.")
    p = config.parameter_count
    tensor_pipeline_parameters = p / (tp * pp)
    model_state = (weight_bytes + 4 + 4 + 4 + 4) * tensor_pipeline_parameters / dp
    largest_group = max(math.prod(s) for s in config.shapes().values())
    padded_sequence = ((sequence + page_size - 1) // page_size) * page_size
    kv_dense = config.layers * batch * padded_sequence * config.heads * (config.content_dim + config.rope_dim + config.value_dim) * weight_bytes
    kv_latent = config.layers * batch * padded_sequence * (config.kv_rank + config.rope_dim) * weight_bytes
    boundary_activation = batch * sequence * config.dim * 4
    # These are scenario inputs, not measurements or named-product specifications.
    payload = tensor_pipeline_parameters * 4
    ring_bytes_per_rank = 2 * (dp - 1) / dp * payload
    ring_seconds = 2 * (dp - 1) * latency_us * 1e-6 + ring_bytes_per_rank / (link_gbps * 1e9 / 8)
    coordinator_seconds_lower_bound = 2 * max(dp - 1, 0) * payload / (link_gbps * 1e9 / 8)
    return {"config": asdict(config), "parameters": p, "ranks": dp * tp * pp,
            "fp32_parameters_bytes": p * 4,
            "ideal_sharded_mixed_model_state_bytes_per_rank": model_state,
            "numpy_fp32_zero_state_bytes_per_rank": 16 * tensor_pipeline_parameters / dp,
            "largest_parameter_materialization_bytes": largest_group * 4,
            "dense_kv_bytes_total": kv_dense, "latent_kv_bytes_total": kv_latent,
            "pipeline_boundary_activation_bytes_per_microbatch": boundary_activation,
            "ring_allreduce_estimated_seconds": ring_seconds,
            "coordinator_link_serialization_lower_bound_seconds": coordinator_seconds_lower_bound,
            "assumed_link_gbps": link_gbps, "assumed_latency_us": latency_us,
            "caveat": "Model-state estimates exclude activations, transient full groups, packing masks, runtime and allocator overhead. Network inputs are illustrative; TCP demo is coordinator routed, not this hypothetical ring."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["tiny", "7b"], default="7b")
    for key in ("dp", "tp", "pp"):
        p.add_argument("--" + key, type=int, default=1)
    p.add_argument("--sequence", type=int, default=2048)
    p.add_argument("--link-gbps", type=float, default=200)
    p.add_argument("--latency-us", type=float, default=5)
    a = p.parse_args()
    print(json.dumps(estimate(Config.seven_b() if a.preset == "7b" else Config(),
          a.dp, a.tp, a.pp, sequence=a.sequence, link_gbps=a.link_gbps, latency_us=a.latency_us), indent=2))


if __name__ == "__main__":
    main()
