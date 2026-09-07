"""Bounded corpus/SFT training with automatic ZeRO-3 and exact local resume.

The parent stages at most 16 MiB of original UTF-8 text/JSONL, trains byte BPE
from at most 64 KiB, and holds a run lease. Workers stream packed samples and
own disjoint positions in a deterministic cycling sequence. The data module can
stream larger sources; this executable deliberately bounds its local workload.
"""
from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import uuid

import numpy as np

from distributed_demo import run_spawned
from rawllm.data import PackedBatch, iter_text, pack_documents, pack_sft
from rawllm.distributed import CollectiveServer, ProcessGroup
from rawllm.model import Config
from rawllm.runtime import RunLease, atomic_json
from rawllm.safeio import read_json, regular_file
from rawllm.sharded_checkpoint import save_sharded_checkpoint, load_sharded_checkpoint, _read_verified_json
from rawllm.tensor import cross_entropy
from rawllm.tokenizer import ByteBPETokenizer
from rawllm.zero_model import ZeroTransformer


ROOT = Path(__file__).resolve().parent
MAX_SOURCE_BYTES = 16 * 1024**2
TOKENIZER_SAMPLE_BYTES = 65536
MAX_RECORD_BYTES = 65536


def digest_file(path: Path, maximum: int, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    consumed = 0
    with regular_file(path, maximum) as source:
        for chunk in iter(lambda: source.read(65536), b""):
            consumed += len(chunk)
            if consumed > maximum:
                raise ValueError("source grew beyond its configured byte bound")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("corpus preparation exceeded the run deadline")
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sft_records(path: Path):
    with regular_file(path, MAX_SOURCE_BYTES) as source:
        while True:
            row = source.readline(MAX_RECORD_BYTES + 1)
            if not row:
                return
            if len(row) > MAX_RECORD_BYTES:
                raise ValueError("SFT JSONL record exceeds 64 KiB")
            if not row.strip():
                continue
            record = json.loads(row.decode("utf8"))
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not isinstance(record.get("response"), str):
                raise ValueError("SFT records require string prompt and response fields")
            yield {"prompt": record["prompt"], "response": record["response"]}


def packed_samples(settings: dict, directory: Path, tokenizer: ByteBPETokenizer):
    source = directory / settings["dataset_file"]
    if settings["mode"] == "pretrain":
        batches = pack_documents(iter_text(source, chunk_bytes=MAX_RECORD_BYTES, max_line_bytes=MAX_RECORD_BYTES),
                                 tokenizer, settings["sequence_length"])
    else:
        batches = pack_sft(sft_records(source), tokenizer, settings["sequence_length"])
    # Excluding unsupervised windows is deterministic on every rank. Prompt
    # tokens within a supervised window remain available as causal context.
    return (batch for batch in batches if batch.supervised_tokens > 0)


class PackedReplay:
    """Replay a bounded-memory, globally ordered cycling stream exactly."""

    def __init__(self, factory, consumed: int = 0, deadline: float | None = None):
        if type(consumed) is not int or consumed < 0:
            raise ValueError("global packed-sample position must be a nonnegative integer")
        self.factory, self.iterator, self.consumed, self.deadline = factory, iter(factory()), 0, deadline
        for _ in range(consumed):
            self.next()

    def next(self) -> PackedBatch:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("packed data replay exceeded the run deadline")
        try:
            value = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.factory())
            try:
                value = next(self.iterator)
            except StopIteration as error:
                raise ValueError("source contains no supervised packed samples") from error
        self.consumed += 1
        return value

    def rank_batch(self, rank: int, world_size: int) -> PackedBatch:
        selected = None
        for owner in range(world_size):
            sample = self.next()
            if owner == rank:
                selected = sample
        return selected


def _counter_template(settings: dict, settings_hash: str, rank: int, step: int, tokens: int) -> dict:
    return {"step": step, "rank": rank, "samples_consumed": step * settings["accumulation"],
            "global_slots_consumed": step * settings["accumulation"] * settings["world_size"],
            "global_target_tokens": tokens, "settings_sha256": settings_hash,
            "dataset_sha256": settings["dataset_sha256"], "tokenizer_sha256": settings["tokenizer_sha256"],
            "config_sha256": settings["config_sha256"]}


def validate_counters(counters: dict, settings: dict, settings_hash: str, rank: int, step: int) -> None:
    expected = _counter_template(settings, settings_hash, rank, step, 0)
    if not isinstance(counters, dict) or set(counters) != set(expected):
        raise ValueError("checkpoint corpus counter fields do not match this run")
    for name in ("step", "rank", "samples_consumed", "global_slots_consumed", "global_target_tokens"):
        if type(counters[name]) is not int or counters[name] < 0:
            raise ValueError("checkpoint corpus counters must be nonnegative integers")
    if any(counters[name] != value for name, value in expected.items() if name != "global_target_tokens"):
        raise ValueError("checkpoint corpus counters, replay position or artifact digests disagree")
    if counters["global_target_tokens"] > step * settings["accumulation"] * settings["world_size"] * settings["sequence_length"]:
        raise ValueError("checkpoint target-token count exceeds the number of consumed positions")


def _inspect_resume(directory: Path, settings: dict, settings_hash: str) -> tuple[int, int]:
    path = directory / "checkpoint" / "manifest.json"
    if not path.exists():
        return 0, 0
    manifest = read_json(path, max_bytes=4 * 1024**2)
    if not isinstance(manifest, dict) or manifest.get("run_id") != settings["run_id"]:
        raise ValueError("checkpoint owner does not match the corpus run")
    step = manifest.get("step")
    if type(step) is not int or not 0 <= step <= 100:
        raise ValueError("checkpoint step exceeds the bounded corpus run")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != settings["world_size"]:
        raise ValueError("checkpoint world size differs from the corpus run")
    token_counts = []
    for rank, descriptor in enumerate(ranks):
        filename = descriptor.get("metadata") if isinstance(descriptor, dict) else None
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("invalid checkpoint rank metadata path")
        metadata = _read_verified_json(path.parent / filename, descriptor["metadata_sha256"], descriptor["metadata_bytes"])
        counters = metadata.get("counters")
        validate_counters(counters, settings, settings_hash, rank, step)
        token_counts.append(counters["global_target_tokens"])
    if len(set(token_counts)) != 1:
        raise ValueError("checkpoint ranks disagree about the consumed target-token count")
    return step, token_counts[0]


def _stage_source(source: Path, destination: Path, deadline: float) -> str:
    partial = destination.with_name("." + destination.name + ".partial-" + uuid.uuid4().hex)
    digest = hashlib.sha256()
    consumed = 0
    try:
        with regular_file(source, MAX_SOURCE_BYTES) as incoming, partial.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(65536), b""):
                consumed += len(block)
                if consumed > MAX_SOURCE_BYTES:
                    raise ValueError("source grew beyond the 16 MiB staging bound")
                if time.monotonic() >= deadline:
                    raise TimeoutError("corpus staging exceeded the run deadline")
                outgoing.write(block)
                digest.update(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return digest.hexdigest()


def _prepare(directory: Path, source: Path | None, mode: str, world_size: int,
             accumulation: int, sequence_length: int, seed: int, learning_rate: float,
             resume: bool, run_id: str, deadline: float) -> tuple[dict, str, int, int]:
    settings_path = directory / "settings.json"
    if settings_path.exists():
        if not resume:
            raise ValueError("corpus output already contains a run; use resume or a new directory")
        settings = read_json(settings_path)
        requested = {"mode": mode, "world_size": world_size, "accumulation": accumulation,
                     "sequence_length": sequence_length, "seed": seed, "learning_rate": learning_rate}
        if any(settings.get(key) != value for key, value in requested.items()):
            raise ValueError("resume settings changed")
        if settings.get("format") != "rawllm-zero-corpus" or settings.get("version") != 1:
            raise ValueError("unsupported corpus run metadata")
        filename = settings.get("dataset_file")
        if filename not in {"source.txt", "source.jsonl"}:
            raise ValueError("invalid staged source filename")
        if digest_file(directory / filename, MAX_SOURCE_BYTES, deadline) != settings["dataset_sha256"]:
            raise ValueError("staged dataset digest changed")
        if source is not None and digest_file(source, MAX_SOURCE_BYTES, deadline) != settings["dataset_sha256"]:
            raise ValueError("provided dataset digest differs from the committed run")
        if digest_file(directory / "tokenizer.json", 32 * 1024**2, deadline) != settings["tokenizer_sha256"]:
            raise ValueError("tokenizer digest changed")
        tokenizer = ByteBPETokenizer.load(directory / "tokenizer.json")
        config = Config(**settings["config"])
        if config.vocab_size != tokenizer.vocab_size or _canonical_hash(settings["config"]) != settings["config_sha256"]:
            raise ValueError("configuration digest or tokenizer vocabulary disagrees")
        if not (directory / "checkpoint" / "manifest.json").is_file():
            raise ValueError("resume requires a committed checkpoint manifest; missing state cannot be reinitialized")
    else:
        if resume:
            raise ValueError("resume requires staged run settings")
        source = source or ROOT / "examples" / ("corpus.txt" if mode == "pretrain" else "sft.jsonl")
        if source.suffix not in {".txt", ".jsonl"} or (mode == "sft" and source.suffix != ".jsonl"):
            raise ValueError("this bounded executable accepts .txt/.jsonl pretraining or .jsonl SFT sources")
        filename = "source" + source.suffix
        dataset_hash = _stage_source(source, directory / filename, deadline)
        if mode == "sft":
            texts = (text for record in sft_records(directory / filename) for text in (record["prompt"], record["response"]))
        else:
            texts = iter_text(directory / filename, chunk_bytes=MAX_RECORD_BYTES, max_line_bytes=MAX_RECORD_BYTES)
        tokenizer = ByteBPETokenizer.train(texts, vocab_size=288, max_bytes=TOKENIZER_SAMPLE_BYTES)
        if time.monotonic() >= deadline:
            raise TimeoutError("tokenizer training exceeded the corpus deadline")
        tokenizer.save(directory / "tokenizer.json")
        config = Config(vocab_size=tokenizer.vocab_size, dim=12, layers=1, heads=2, q_rank=6, kv_rank=4,
                        content_dim=4, rope_dim=4, value_dim=4, hidden_dim=24, max_seq_len=64)
        settings = {"format": "rawllm-zero-corpus", "version": 1, "run_id": run_id,
                    "mode": mode, "world_size": world_size, "accumulation": accumulation,
                    "sequence_length": sequence_length, "seed": seed, "learning_rate": learning_rate,
                    "config": asdict(config), "config_sha256": _canonical_hash(asdict(config)),
                    "dataset_file": filename, "dataset_sha256": dataset_hash, "source_original": str(source.resolve()),
                    "tokenizer_sha256": digest_file(directory / "tokenizer.json", 32 * 1024**2, deadline),
                    "max_source_bytes": MAX_SOURCE_BYTES, "tokenizer_sample_bytes": TOKENIZER_SAMPLE_BYTES,
                    "numpy_version": np.__version__, "packing": "document-isolated causal windows; global strided replay"}
        atomic_json(settings_path, settings)
    settings_hash = digest_file(settings_path, 1024**2, deadline)
    step, tokens = _inspect_resume(directory, settings, settings_hash)
    return settings, settings_hash, step, tokens


def _train_rank(rank: int, host: str, port: int, token: str, directory_text: str,
                settings: dict, settings_hash: str, target_steps: int, deadline: float) -> dict:
    directory = Path(directory_text)
    world_size = settings["world_size"]
    tokenizer = ByteBPETokenizer.load(directory / "tokenizer.json")
    config = Config(**settings["config"])
    with ProcessGroup(rank, world_size, host, port, token, timeout=12) as group:
        model = ZeroTransformer(config, group, seed=settings["seed"], learning_rate=settings["learning_rate"])
        rng = np.random.default_rng(settings["seed"] + rank)
        checkpoint = directory / "checkpoint"
        counters = _counter_template(settings, settings_hash, rank, 0, 0)
        if (checkpoint / "manifest.json").exists():
            counters = load_sharded_checkpoint(checkpoint, model.optimizer, rng, expected_run_id=settings["run_id"])
            validate_counters(counters, settings, settings_hash, rank, model.optimizer.step_number)
        else:
            save_sharded_checkpoint(checkpoint, model.optimizer, rng, counters, run_id=settings["run_id"])
        stream = PackedReplay(lambda: packed_samples(settings, directory, tokenizer),
                              counters["global_slots_consumed"], deadline)
        losses, token_masses = [], []
        start_step = counters["step"]
        while counters["step"] < target_steps:
            stop = group.all_reduce(np.array([int(time.monotonic() >= deadline)], dtype=np.int64))[0]
            if stop:
                break
            batches = [stream.rank_batch(rank, world_size) for _ in range(settings["accumulation"])]
            local_mass = sum(batch.supervised_tokens for batch in batches)
            global_mass = int(group.all_reduce(np.array([local_mass], dtype=np.int64))[0])
            if global_mass <= 0:
                raise ValueError("global batch contains no supervised targets")
            model.zero_grad()
            local_numerator = 0.0
            for batch in batches:
                numerator = cross_entropy(model(batch.input_ids, batch.attention_mask), batch.targets,
                                          mask=batch.loss_mask, reduction="sum")
                local_numerator += numerator.item()
                # accumulate_gradient averages data ranks. Multiplying each
                # local summed-token loss by D/M cancels that mean and produces
                # (sum_r sum_tokens gradient)/global_supervised_token_mass.
                (numerator * (world_size / global_mass)).backward()
                del numerator
            mean_loss = group.all_reduce(np.array([local_numerator], dtype=np.float64))[0] / global_mass
            report = model.step(max_grad_norm=1.0)
            if not report["updated"] or not math.isfinite(mean_loss):
                raise FloatingPointError("corpus update was nonfinite; last committed checkpoint is intact")
            counters = _counter_template(settings, settings_hash, rank, counters["step"] + 1,
                                         counters["global_target_tokens"] + global_mass)
            if stream.consumed != counters["global_slots_consumed"]:
                raise AssertionError("packed sample replay position drifted")
            save_sharded_checkpoint(checkpoint, model.optimizer, rng, counters, run_id=settings["run_id"])
            losses.append(float(mean_loss))
            token_masses.append({"local": local_mass, "global": global_mass})
        group.barrier()
        return {"rank": rank, "starting_step": start_step, "step": counters["step"],
                "global_target_tokens": counters["global_target_tokens"], "losses": losses,
                "token_masses": token_masses, "memory": model.memory_report()}


def run_training(output: str | Path, *, mode: str = "pretrain", source: str | Path | None = None,
                 steps: int = 2, world_size: int = 2, accumulation: int = 2,
                 sequence_length: int = 16, seed: int = 17, learning_rate: float = 3e-4,
                 resume: bool = False, max_seconds: float = 30) -> dict:
    started, directory = time.monotonic(), Path(output).resolve()
    if mode not in {"pretrain", "sft"} or type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("choose pretrain/SFT and 1..100 target steps")
    if (any(type(value) is not int for value in (world_size, accumulation, sequence_length, seed))
            or not 1 <= world_size <= 4 or not 1 <= accumulation <= 4 or not 4 <= sequence_length <= 64
            or not 0 <= seed < 2**32):
        raise ValueError("use 1..4 ranks/microbatches and 4..64 sequence positions")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("use a finite positive learning rate and 0..120 second deadline")
    deadline = started + max_seconds
    source = None if source is None else Path(source).resolve()
    with RunLease(directory) as lease:
        settings, settings_hash, committed_step, committed_tokens = _prepare(
            directory, source, mode, world_size, accumulation, sequence_length, seed, learning_rate,
            resume, lease.run_id, deadline)
        if committed_step > steps:
            raise ValueError("target steps precede the committed checkpoint")
        atomic_json(directory / "status.json", {"status": "running", "run_id": settings["run_id"], "target_steps": steps})
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("corpus preparation exhausted the run budget")
            with CollectiveServer(world_size, timeout=12) as server:
                ranks = run_spawned(_train_rank, world_size,
                                    (server.host, server.port, server.token, str(directory), settings,
                                     settings_hash, steps, deadline), timeout=remaining)
            if len({rank["step"] for rank in ranks}) != 1 or len({rank["global_target_tokens"] for rank in ranks}) != 1:
                raise AssertionError("ranks completed different corpus training positions")
            report = {"mode": mode, "training": "actual packed corpus with automatic ZeRO-3", "world_size": world_size,
                      "parameters": Config(**settings["config"]).parameter_count,
                      "starting_step": committed_step, "completed_steps": ranks[0]["step"], "target_steps": steps,
                      "global_target_tokens": ranks[0]["global_target_tokens"], "losses": ranks[0]["losses"],
                      "token_mass_by_rank": [rank["token_masses"] for rank in ranks],
                      "memory_by_rank": [rank["memory"] for rank in ranks],
                      "dataset_sha256": settings["dataset_sha256"], "tokenizer_sha256": settings["tokenizer_sha256"],
                      "settings_sha256": settings_hash, "config_sha256": settings["config_sha256"],
                      "checkpoint": str(directory / "checkpoint"), "run_directory": str(directory),
                      "elapsed_seconds": time.monotonic() - started,
                      "status": "completed" if ranks[0]["step"] == steps else "budget_reached",
                      "source_byte_limit": MAX_SOURCE_BYTES, "tokenizer_training_byte_limit": TOKENIZER_SAMPLE_BYTES,
                      "scope": "pretraining/SFT with exact fixed-topology resume; no DPO or train.py-compatible dense export"}
            atomic_json(directory / "status.json", report)
            return report
        except BaseException as error:
            atomic_json(directory / "status.json", {"status": "failed", "run_id": settings["run_id"],
                                                    "error_type": type(error).__name__})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pretrain", "sft"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--steps", type=int, default=2, help="target completed updates, including restored updates")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    def terminate(signum, frame):
        # Raising on the parent unwinds run_spawned's owned-worker cleanup and
        # the run lease. The most recent published checkpoint stays committed.
        raise KeyboardInterrupt("corpus training received SIGTERM")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        report = run_training(options.output, mode=options.mode, source=options.data, steps=options.steps,
                              world_size=options.world_size, accumulation=options.accumulation,
                              sequence_length=options.sequence_length, seed=options.seed,
                              learning_rate=options.learning_rate, resume=options.resume, max_seconds=options.max_seconds)
    finally:
        signal.signal(signal.SIGTERM, previous)
    if options.report:
        atomic_json(options.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
