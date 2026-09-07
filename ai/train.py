#!/usr/bin/env python3
"""Complete bounded local pretraining, SFT and DPO entry point using NumPy only."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from rawllm.model import Config, Transformer
from rawllm.tensor import no_grad
from rawllm.tokenizer import ByteBPETokenizer
from rawllm.data import iter_text, pack_documents, pack_sft, preference_batch
from rawllm.alignment import sft_loss, sequence_log_probs, dpo_loss
from rawllm.optim import AdamW, clip_grad_norm, DynamicLossScaler, save_checkpoint, load_checkpoint
from rawllm.runtime import RunLease, StopRequest, atomic_json, reconcile_log
from rawllm.safeio import read_json, read_npz

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_records(path):
    with Path(path).open("rb") as f:
        while True:
            line = f.readline(1024 * 1024 + 1)
            if not line:
                break
            if len(line) > 1024 * 1024:
                raise ValueError("Training JSONL row exceeds 1 MiB limit.")
            if line.strip():
                yield json.loads(line)


class ReplayStream:
    """Bounded-memory cycling stream with exact deterministic replay on resume."""
    def __init__(self, factory, consumed=0, should_stop=None):
        if type(consumed) is not int or consumed < 0:
            raise ValueError("Replay position must be a nonnegative integer")
        self.factory, self.iterator, self.consumed = factory, iter(factory()), 0
        for _ in range(consumed):
            if should_stop is not None and should_stop():
                raise TimeoutError("Stream replay interrupted; the committed checkpoint remains unchanged")
            self.next()

    def next(self):
        try:
            value = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.factory())
            try:
                value = next(self.iterator)
            except StopIteration as exc:
                raise ValueError("Dataset yielded no supervised training samples.") from exc
        self.consumed += 1
        return value


def batch_loss(model, batch):
    logits = model(batch.input_ids, batch.attention_mask)
    return sft_loss(logits, batch.targets, batch.loss_mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pretrain", "sft", "dpo"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, help="Existing run whose model/tokenizer initialize a new phase")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=3, help="Target optimizer attempts, including resumed steps")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=.1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100000 or not 1 <= args.accumulation <= 32 or not 4 <= args.sequence_length <= 256:
        parser.error("Use positive bounded steps, accumulation 1..32, and length 4..256.")
    if not 0 < args.max_seconds <= 600:
        parser.error("Use a positive time budget up to 600 seconds.")
    if args.resume and args.initialize:
        parser.error("Resume and initialize are mutually exclusive.")
    with RunLease(args.output) as lease, StopRequest() as stop:
        if (args.output / "settings.json").exists() and not args.resume:
            raise ValueError("Output already contains a run. Choose a new path or --resume.")
        status = {"run_id": lease.run_id, "status": "running", "pid": os.getpid()}
        atomic_json(args.output / "status.json", status)
        try:
            result = run_training(args, stop)
        except BaseException as error:
            atomic_json(args.output / "status.json", {**status, "status": "failed", "error_type": type(error).__name__})
            raise
        finished = "stopped" if stop.requested else ("budget_reached" if result["budget_reached"] and result["step"] < args.steps else "completed")
        atomic_json(args.output / "status.json", {**status, **result, "status": finished})


def run_training(args, stop):
    deadline = time.monotonic() + args.max_seconds
    default_data = {"pretrain": "corpus.txt", "sft": "sft.jsonl", "dpo": "preferences.jsonl"}
    source = (args.data or ROOT / "examples" / default_data[args.mode]).resolve()
    dataset_hash = digest(source)
    output = args.output.resolve()
    if (output / "settings.json").exists() and not args.resume:
        raise ValueError("Output already contains a run. Choose a new path or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    inherited = output if args.resume else args.initialize
    if inherited:
        settings_in = read_json(inherited / "settings.json")
        if digest(inherited / "tokenizer.json") != settings_in["tokenizer_sha256"]:
            raise ValueError("Tokenizer checksum differs from the saved run.")
        tokenizer = ByteBPETokenizer.load(inherited / "tokenizer.json")
        config = Config(**settings_in["config"])
    else:
        tokenizer = ByteBPETokenizer.train(iter_text(ROOT / "examples/corpus.txt"), vocab_size=288, max_bytes=65536)
        config = Config(vocab_size=tokenizer.vocab_size, max_seq_len=256)
    if args.sequence_length > config.max_seq_len:
        raise ValueError("Sequence length exceeds inherited model configuration.")
    model = Transformer(config, args.seed)
    model.precision = args.precision
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scaler = DynamicLossScaler(initial_scale=128 if args.precision == "fp16" else 1)
    counters = {"step": 0, "samples_consumed": 0, "updates": 0, "skipped_updates": 0}
    if inherited:
        loaded = load_checkpoint(inherited / "checkpoint", model.parameters(), optimizer, rng, scaler)
        if args.resume:
            required = {"mode": args.mode, "dataset_sha256": dataset_hash, "precision": args.precision,
                        "sequence_length": args.sequence_length, "accumulation": args.accumulation,
                        "beta": args.beta}
            if any(settings_in.get(k) != v for k, v in required.items()):
                raise ValueError("Resume settings or source digest changed; initialize a new phase instead.")
            counters = loaded
            if set(counters) != {"step", "samples_consumed", "updates", "skipped_updates"} or any(type(n) is not int or n < 0 for n in counters.values()):
                raise ValueError("Invalid training counters in checkpoint")
            if counters["step"] != counters["updates"] + counters["skipped_updates"] or counters["samples_consumed"] != counters["step"] * args.accumulation:
                raise ValueError("Training counters do not describe complete optimizer attempts")
        else:
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            scaler = DynamicLossScaler(initial_scale=128 if args.precision == "fp16" else 1)
            rng = np.random.default_rng(args.seed)
    tokenizer.save(output / "tokenizer.json")
    settings = {"config": asdict(config), "mode": args.mode, "dataset": str(source),
                "dataset_sha256": dataset_hash, "precision": args.precision,
                "sequence_length": args.sequence_length, "accumulation": args.accumulation,
                "beta": args.beta, "seed": settings_in["seed"] if args.resume else args.seed, "learning_rate": optimizer.lr,
                "tokenizer_sha256": digest(output / "tokenizer.json"),
                "numpy_version": np.__version__, "initialization": str(inherited) if inherited else "random"}
    reference = None
    if args.mode == "dpo":
        reference = Transformer(config, args.seed)
        reference.precision = args.precision
        refpath = output / "reference.npz"
        if args.resume:
            if digest(refpath) != settings_in["reference_sha256"]:
                raise ValueError("Frozen reference checksum mismatch.")
            arrays = read_npz(refpath, {name: (p.shape, p.data.dtype) for name, p in reference.params.items()},
                              sha256=settings_in["reference_sha256"])
            for name, parameter in reference.params.items():
                parameter.data[...] = arrays[name]
        else:
            for name, parameter in reference.params.items():
                parameter.data[...] = model.params[name].data
            np.savez(refpath, **{name: p.data for name, p in reference.params.items()})
        settings["reference_sha256"] = digest(refpath)
    atomic_json(output / "settings.json", settings)
    if args.mode == "pretrain":
        factory = lambda: (b for b in pack_documents(iter_text(source), tokenizer, args.sequence_length) if b.loss_mask.sum() > 0)
    elif args.mode == "sft":
        factory = lambda: (b for b in pack_sft(json_records(source), tokenizer, args.sequence_length) if b.loss_mask.sum() > 0)
    else:
        factory = lambda: ((preference_batch(r["prompt"], r["chosen"], tokenizer, args.sequence_length),
                            preference_batch(r["prompt"], r["rejected"], tokenizer, args.sequence_length)) for r in json_records(source))
    stream = ReplayStream(factory, counters["samples_consumed"],
                          should_stop=lambda: stop.requested or time.monotonic() >= deadline)
    if args.resume:
        reconcile_log(output / "training.jsonl", counters["step"])
    mode = "a" if args.resume else "w"
    with (output / "training.jsonl").open(mode) as log:
        while counters["step"] < args.steps and time.monotonic() < deadline and not stop.requested:
            batches = [stream.next() for _ in range(args.accumulation)]
            normalizer = args.accumulation if args.mode == "dpo" else sum(float(b.loss_mask.sum()) for b in batches)
            optimizer.zero_grad()
            total_loss = 0.0
            forward_error = None
            try:
                for batch in batches:
                    if args.mode == "dpo":
                        chosen, rejected = batch
                        policy_chosen = sequence_log_probs(model(chosen.input_ids, chosen.attention_mask), chosen.targets, chosen.loss_mask)
                        policy_rejected = sequence_log_probs(model(rejected.input_ids, rejected.attention_mask), rejected.targets, rejected.loss_mask)
                        with no_grad():
                            ref_chosen = sequence_log_probs(reference(chosen.input_ids, chosen.attention_mask), chosen.targets, chosen.loss_mask)
                            ref_rejected = sequence_log_probs(reference(rejected.input_ids, rejected.attention_mask), rejected.targets, rejected.loss_mask)
                        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta) / normalizer
                    else:
                        loss = batch_loss(model, batch) * (float(batch.loss_mask.sum()) / normalizer)
                    total_loss += float(loss.data)
                    scaler.scale(loss).backward()
            except FloatingPointError as exc:
                forward_error = str(exc)
            finite = forward_error is None and scaler.unscale_(model.parameters()) and np.isfinite(total_loss)
            gradient_norm = None
            if finite:
                gradient_norm = clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                counters["updates"] += 1
            else:
                counters["skipped_updates"] += 1
            scaler.update(bool(finite))
            counters["step"] += 1
            counters["samples_consumed"] = stream.consumed
            optimizer.zero_grad()
            record = {**counters, "loss": total_loss if np.isfinite(total_loss) else None,
                      "gradient_norm": gradient_norm, "scale": scaler.scale_value, "finite": bool(finite), "forward_error": forward_error}
            save_checkpoint(output / "checkpoint", model.parameters(), optimizer, rng, counters, scaler)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            print(json.dumps(record), flush=True)
            if forward_error is not None:
                raise FloatingPointError("Stopped after saving unchanged weights: " + forward_error)
    if counters["step"] == 0:
        save_checkpoint(output / "checkpoint", model.parameters(), optimizer, rng, counters, scaler)
    result = {"output": str(output), "parameters": config.parameter_count, **counters,
              "stop_requested": stop.requested, "budget_reached": time.monotonic() >= deadline}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
