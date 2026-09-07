"""Collective, durable checkpoints for fixed-topology sharded AdamW jobs.

All participating ranks require the same trusted, shared POSIX directory. Each
rank writes an immutable numerical archive and metadata file, fsyncs both, and
acknowledges preparation. The leader verifies every prepared shard before one
atomic manifest rename publishes the generation. A failure before that rename
leaves the previous committed generation intact. A connection failure after the
rename can make the caller uncertain about success; loading the manifest resolves
that uncertainty. Orphan prepared generations are never considered committed.

A nonblocking advisory writer lock protects each transaction. Persistent run IDs
prevent an unrelated job from replacing an existing run. Load restores that ID
on the optimizer so continuation may save without a separate ownership argument.
These are local POSIX durability guarantees, not a distributed filesystem lease
or protection against another process maliciously modifying trusted directories.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Mapping
import uuid

import numpy as np

from .safeio import read_json, read_npz, regular_file
from .zero import ShardedAdamW, shard_bounds


_ARRAYS = ("parameters", "gradients", "first_moment", "second_moment")
_METADATA_LIMIT = 1_048_576
_MANIFEST_LIMIT = 4_194_304
_WIRE_LIMIT = 131_072
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ShardedCheckpointError(RuntimeError):
    """Checkpoint preparation or validation failed collectively."""


def _encode(value) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.ndim > 16:
            raise ValueError("Checkpoint JSON arrays must contain plain real numeric values")
        return {"__array__": value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Checkpoint JSON mapping keys must be strings")
        if set(value) == {"__array__", "dtype"}:
            raise ValueError("Checkpoint mapping collides with the reserved numerical array tag")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Checkpoint numerical scalars must be finite")
    if isinstance(value, dict):
        if set(value) == {"__array__", "dtype"}:
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biuf" or dtype.hasobject or dtype.fields or dtype.subdtype:
                raise ValueError("Invalid JSON numerical array dtype")
            array = np.asarray(value["__array__"], dtype=dtype)
            if array.ndim > 16 or not np.isfinite(array).all():
                raise ValueError("Invalid JSON numerical array contents")
            return array
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _json_bytes(value, limit: int) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > limit:
        raise ValueError("Checkpoint JSON exceeds its configured byte limit")
    return encoded


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate checkpoint JSON field")
        result[name] = value
    return result


def _read_verified_json(path: Path, digest: str, size: int) -> dict:
    if type(size) is not int or not 0 < size <= _METADATA_LIMIT or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("Invalid metadata size or digest")
    with regular_file(path, _METADATA_LIMIT) as handle:
        raw = handle.read(_METADATA_LIMIT + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Rank metadata checksum or size mismatch")
    def reject_constant(value):
        raise ValueError("Non-finite checkpoint JSON constant")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Checkpoint JSON exponent exceeds finite numerical range")
        return number
    value = json.loads(raw, object_pairs_hook=_unique_fields, parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    return value


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(path: Path, maximum: int) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    with regular_file(path, maximum) as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            count += len(block)
            if count > maximum:
                raise ValueError("Checkpoint payload grew beyond its byte limit")
            digest.update(block)
    return digest.hexdigest(), count


def _collective(group, phase: str, operation):
    """Gather successes and bounded errors so local I/O failure cannot strand peers."""
    try:
        result = {"ok": True, "value": operation()}
        raw = _json_bytes(result, _WIRE_LIMIT)
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"[:4096]}
        raw = _json_bytes(result, _WIRE_LIMIT)
    gathered = group.all_gather(np.frombuffer(raw, dtype=np.uint8))
    messages = []
    for item in gathered:
        if item.dtype != np.uint8 or item.ndim != 1 or item.size > _WIRE_LIMIT:
            raise ShardedCheckpointError(f"{phase}: malformed collective acknowledgement")
        message = json.loads(item.tobytes())
        if not isinstance(message, dict) or type(message.get("ok")) is not bool:
            raise ShardedCheckpointError(f"{phase}: invalid collective acknowledgement")
        messages.append(message)
    errors = [f"rank {group.members[index]}: {message.get('error', 'unknown failure')}"
              for index, message in enumerate(messages) if not message["ok"]]
    if errors:
        raise ShardedCheckpointError(f"{phase}: " + "; ".join(errors))
    return [message["value"] for message in messages]


def _layout(optimizer: ShardedAdamW) -> dict:
    return {"names": list(optimizer.partitions),
            "shapes": [list(partition.shape) for partition in optimizer.partitions.values()],
            "learning_rate": optimizer.learning_rate, "betas": list(optimizer.betas),
            "epsilon": optimizer.epsilon, "weight_decay": optimizer.weight_decay,
            "gradient_reduction": optimizer.gradient_reduction}


def _layout_digest(optimizer: ShardedAdamW) -> str:
    return hashlib.sha256(_json_bytes(_layout(optimizer), _METADATA_LIMIT)).hexdigest()


def _optimizer_metadata(optimizer: ShardedAdamW) -> dict:
    return {**_layout(optimizer), "rank": optimizer.group.group_rank,
            "world_size": optimizer.group.size, "step": optimizer.step_number,
            "accumulations": dict(optimizer._accumulations)}


def _check_active(optimizer: ShardedAdamW, maximum: int, validate_values: bool = True) -> None:
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_payload_bytes must be a positive integer")
    if optimizer._active_name is not None:
        raise ValueError("Release materialized tensors before checkpointing")
    if optimizer.group.size > 1024:
        raise ValueError("Checkpoint manifest supports at most 1024 ranks")
    local_size = 0
    for name, partition in optimizer.partitions.items():
        total = math.prod(partition.shape)
        start, stop = shard_bounds(total, optimizer.group.group_rank, optimizer.group.size)
        if (partition.name != name or partition.total_size != total
                or (partition.global_start, partition.global_stop) != (start, stop)
                or (partition.local_start, partition.local_stop) != (local_size, local_size + stop - start)):
            raise ValueError("Live optimizer partition coordinates are inconsistent")
        local_size += stop - start
    if optimizer.parameters.shape != (local_size,):
        raise ValueError("Live optimizer arrays do not match the fixed partition size")
    if 4 * optimizer.parameters.nbytes > maximum:
        raise MemoryError("Local sharded state exceeds checkpoint payload budget")
    if validate_values and (type(optimizer.step_number) is not int or optimizer.step_number < 0):
        raise ValueError("Invalid live optimizer step")
    if validate_values and (set(optimizer._accumulations) != set(optimizer.partitions) or any(type(n) is not int or n < 0 for n in optimizer._accumulations.values())):
        raise ValueError("Invalid live optimizer accumulation counters")
    for name in _ARRAYS:
        array = getattr(optimizer, name)
        if array.shape != optimizer.parameters.shape or array.dtype != np.float32 or (validate_values and not np.isfinite(array).all()):
            raise ValueError(f"Invalid live optimizer array: {name}")
        if not validate_values and not array.flags.writeable:
            raise ValueError(f"Live optimizer destination is read-only: {name}")
    if validate_values and np.any(optimizer.second_moment < 0):
        raise ValueError("Adam second moments cannot be negative")


def _validate_identifier(value: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("Run IDs must contain 1..128 ASCII letters, digits, periods, underscores or hyphens")


def _owners(optimizer: ShardedAdamW) -> dict:
    if not hasattr(optimizer, "_checkpoint_owners"):
        optimizer._checkpoint_owners = {}
    return optimizer._checkpoint_owners


def _write_immutable(path: Path, writer) -> None:
    temporary = path.with_name("." + path.name + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # link() refuses an existing destination, preserving immutability even
        # when a caller accidentally reuses a generation name.
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _validate_manifest(manifest: dict, optimizer: ShardedAdamW) -> None:
    required = {"format", "version", "run_id", "generation", "topology", "layout_sha256", "step", "ranks"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Invalid sharded checkpoint manifest fields")
    if manifest["format"] != "rawllm-sharded-checkpoint" or type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("Unsupported sharded checkpoint format")
    _validate_identifier(manifest["run_id"])
    if not isinstance(manifest["generation"], str) or not re.fullmatch(r"[0-9a-f]{32}", manifest["generation"]):
        raise ValueError("Invalid checkpoint generation")
    group = optimizer.group
    topology = manifest["topology"]
    if (not isinstance(topology, dict) or set(topology) != {"world_size", "members"}
            or type(topology["world_size"]) is not int or not isinstance(topology["members"], list)
            or any(type(rank) is not int for rank in topology["members"])):
        raise ValueError("Invalid checkpoint topology schema")
    if manifest["topology"] != {"world_size": group.world_size, "members": list(group.members)}:
        raise ValueError("Checkpoint fixed topology differs from current process group")
    if manifest["layout_sha256"] != _layout_digest(optimizer):
        raise ValueError("Checkpoint parameter layout or optimizer hyperparameters differ")
    if type(manifest["step"]) is not int or manifest["step"] < 0:
        raise ValueError("Invalid checkpoint step")
    if not isinstance(manifest["ranks"], list) or len(manifest["ranks"]) != group.size:
        raise ValueError("Incomplete rank manifest")
    fields = {"rank", "global_rank", "payload", "payload_sha256", "payload_bytes",
              "metadata", "metadata_sha256", "metadata_bytes"}
    for rank, descriptor in enumerate(manifest["ranks"]):
        if not isinstance(descriptor, dict) or set(descriptor) != fields:
            raise ValueError("Invalid rank descriptor fields")
        if (type(descriptor["rank"]) is not int or descriptor["rank"] != rank
                or type(descriptor["global_rank"]) is not int or descriptor["global_rank"] != group.members[rank]):
            raise ValueError("Rank descriptor topology mismatch")
        stem = f"generation-{manifest['generation']}-rank-{group.members[rank]}"
        if descriptor["payload"] != stem + ".npz" or descriptor["metadata"] != stem + ".json":
            raise ValueError("Rank descriptor contains an invalid path")
        for field in ("payload_sha256", "metadata_sha256"):
            if not isinstance(descriptor[field], str) or not _SHA256.fullmatch(descriptor[field]):
                raise ValueError("Invalid rank artifact digest")
        if type(descriptor["payload_bytes"]) is not int or descriptor["payload_bytes"] < 1:
            raise ValueError("Invalid payload byte count")
        if type(descriptor["metadata_bytes"]) is not int or not 0 < descriptor["metadata_bytes"] <= _METADATA_LIMIT:
            raise ValueError("Invalid metadata byte count")


def _read_rank(directory: Path, manifest: dict, optimizer: ShardedAdamW,
               rank: int, maximum: int):
    descriptor = manifest["ranks"][rank]
    metadata = _read_verified_json(directory / descriptor["metadata"], descriptor["metadata_sha256"], descriptor["metadata_bytes"])
    if set(metadata) != {"version", "generation", "run_id", "optimizer", "rng", "counters"}:
        raise ValueError("Invalid rank metadata fields")
    if type(metadata["version"]) is not int or metadata["version"] != 1 or metadata["generation"] != manifest["generation"] or metadata["run_id"] != manifest["run_id"]:
        raise ValueError("Rank metadata generation or owner mismatch")
    state = metadata["optimizer"]
    expected = {**_layout(optimizer), "rank": rank, "world_size": optimizer.group.size,
                "step": manifest["step"]}
    if not isinstance(state, dict) or set(state) != set(expected) | {"accumulations"}:
        raise ValueError("Invalid rank optimizer metadata fields")
    if any(type(state[name]) is not int for name in ("rank", "world_size", "step")):
        raise ValueError("Rank optimizer indices must be integer values")
    if any(state[key] != value for key, value in expected.items()):
        raise ValueError("Rank optimizer metadata disagrees with fixed layout")
    accumulations = state["accumulations"]
    if not isinstance(accumulations, dict) or set(accumulations) != set(optimizer.partitions) or any(type(n) is not int or n < 0 for n in accumulations.values()):
        raise ValueError("Invalid rank gradient accumulation counters")
    local_size = sum(shard_bounds(math.prod(partition.shape), rank, optimizer.group.size)[1]
                     - shard_bounds(math.prod(partition.shape), rank, optimizer.group.size)[0]
                     for partition in optimizer.partitions.values())
    payload = directory / descriptor["payload"]
    # Bounds are derived from the live optimizer layout, never from an archive's
    # claimed shape. safeio checks every NPY header before allocating arrays.
    arrays = read_npz(payload, {name: ((local_size,), np.float32) for name in _ARRAYS},
                      sha256=descriptor["payload_sha256"], max_bytes=maximum)
    with regular_file(payload, maximum + 81_920) as handle:
        if os.fstat(handle.fileno()).st_size != descriptor["payload_bytes"]:
            raise ValueError("Payload byte count does not match manifest")
    if np.any(arrays["second_moment"] < 0):
        raise ValueError("Adam second moments cannot be negative")
    if not isinstance(metadata["counters"], dict) or not isinstance(metadata["rng"], dict):
        raise ValueError("Checkpoint counters and RNG state must be objects")
    rng_state, counters = _decode(metadata["rng"]), _decode(metadata["counters"])
    if not isinstance(rng_state, dict) or not isinstance(counters, dict):
        raise ValueError("Decoded checkpoint RNG state and counters must remain mappings")
    return {**state, **arrays}, rng_state, counters


def save_sharded_checkpoint(path: str | Path, optimizer: ShardedAdamW,
                            rng: np.random.Generator, counters: Mapping, *,
                            run_id: str | None = None, max_payload_bytes: int = 1024 ** 3) -> Path:
    """Collectively save one generation; every group rank must call in order.

    Per-rank RNG and counters may differ. Layout, optimizer step, and accumulated
    microbatch counts must agree across ranks. No optimizer state is mutated.
    """
    group, directory = optimizer.group, Path(path).absolute()
    owner_key, leader = str(directory), group.group_rank == 0
    lock_descriptor = None
    def preflight():
        _check_active(optimizer, max_payload_bytes)
        if directory.is_symlink():
            raise ValueError("Checkpoint directory cannot be a symbolic link")
        if not isinstance(counters, Mapping):
            raise TypeError("Checkpoint counters must be a mapping")
        selected_owner = run_id if run_id is not None else _owners(optimizer).get(owner_key)
        if selected_owner is not None:
            _validate_identifier(selected_owner)
        _json_bytes({"rng": _encode(rng.bit_generator.state), "counters": _encode(counters)}, _METADATA_LIMIT)
        return {"directory": owner_key, "owner": selected_owner, "layout": _layout_digest(optimizer),
                "step": optimizer.step_number, "accumulations": optimizer._accumulations}
    readiness = _collective(group, "checkpoint preflight", preflight)
    if any(item != readiness[0] for item in readiness[1:]):
        raise ShardedCheckpointError("Ranks disagree about checkpoint destination, owner, layout or training position")
    try:
        def acquire():
            nonlocal lock_descriptor
            if not leader:
                return None
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink():
                raise ValueError("Checkpoint directory cannot be a symbolic link")
            # Persist the new checkpoint directory's entry as well as the
            # generation files written inside it later in this transaction.
            _sync_directory(directory.parent)
            lock_descriptor = os.open(directory / ".writer.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise ValueError("Writer lock must be a regular file")
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            chosen = readiness[0]["owner"] or uuid.uuid4().hex
            committed = directory / "manifest.json"
            if committed.exists() or committed.is_symlink():
                previous = read_json(committed, max_bytes=_MANIFEST_LIMIT)
                _validate_manifest(previous, optimizer)
                if previous["run_id"] != chosen:
                    raise ValueError("Checkpoint directory belongs to another run; load its committed state or choose a new directory")
            return {"run_id": chosen, "generation": uuid.uuid4().hex}
        transaction = _collective(group, "checkpoint writer ownership", acquire)[0]
        manifest = {"format": "rawllm-sharded-checkpoint", "version": 1, **transaction,
                    "topology": {"world_size": group.world_size, "members": list(group.members)},
                    "layout_sha256": _layout_digest(optimizer), "step": optimizer.step_number}
        def prepare():
            stem = f"generation-{transaction['generation']}-rank-{group.rank}"
            payload, metadata_path = directory / (stem + ".npz"), directory / (stem + ".json")
            _write_immutable(payload, lambda handle: np.savez(handle, **{name: getattr(optimizer, name) for name in _ARRAYS}))
            payload_digest, payload_bytes = _digest(payload, max_payload_bytes + 81_920)
            metadata = {"version": 1, **transaction, "optimizer": _optimizer_metadata(optimizer),
                        "rng": _encode(rng.bit_generator.state), "counters": _encode(counters)}
            encoded = _json_bytes(metadata, _METADATA_LIMIT)
            _write_immutable(metadata_path, lambda handle: handle.write(encoded))
            _sync_directory(directory)
            return {"rank": group.group_rank, "global_rank": group.rank,
                    "payload": payload.name, "payload_sha256": payload_digest, "payload_bytes": payload_bytes,
                    "metadata": metadata_path.name, "metadata_sha256": hashlib.sha256(encoded).hexdigest(), "metadata_bytes": len(encoded)}
        manifest["ranks"] = _collective(group, "checkpoint durable rank preparation", prepare)
        def commit():
            if not leader:
                return None
            _validate_manifest(manifest, optimizer)
            for rank in range(group.size):
                _read_rank(directory, manifest, optimizer, rank, max_payload_bytes)
            encoded = _json_bytes(manifest, _MANIFEST_LIMIT)
            temporary = directory / (".manifest-" + transaction["generation"] + ".partial")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, directory / "manifest.json")
                _sync_directory(directory)
            finally:
                os.close(descriptor)
                if temporary.exists():
                    temporary.unlink()
            return transaction["generation"]
        _collective(group, "checkpoint manifest commit", commit)
        _owners(optimizer)[owner_key] = transaction["run_id"]
        return directory / "manifest.json"
    finally:
        if lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)


def load_sharded_checkpoint(path: str | Path, optimizer: ShardedAdamW,
                            rng: np.random.Generator, *, expected_run_id: str | None = None,
                            max_payload_bytes: int = 1024 ** 3) -> dict:
    """Validate all shards collectively before restoring any live rank state.

    Restart requires identical global world size, group membership, tensor layout
    and optimizer hyperparameters. No resharding or elastic membership is implied.
    """
    group, directory = optimizer.group, Path(path).absolute()
    manifest = None
    def inspect():
        nonlocal manifest
        _check_active(optimizer, max_payload_bytes, validate_values=False)
        if directory.is_symlink():
            raise ValueError("Checkpoint directory cannot be a symbolic link")
        manifest = read_json(directory / "manifest.json", max_bytes=_MANIFEST_LIMIT)
        _validate_manifest(manifest, optimizer)
        if expected_run_id is not None:
            _validate_identifier(expected_run_id)
            if manifest["run_id"] != expected_run_id:
                raise ValueError("Committed checkpoint belongs to a different run")
        return {"directory": str(directory), "manifest_sha256": hashlib.sha256(_json_bytes(manifest, _MANIFEST_LIMIT)).hexdigest()}
    inspections = _collective(group, "checkpoint committed manifest validation", inspect)
    if any(item != inspections[0] for item in inspections[1:]):
        raise ShardedCheckpointError("Ranks observed different committed checkpoint manifests")
    restored = None
    def validate_local():
        nonlocal restored
        state, rng_state, counters = _read_rank(directory, manifest, optimizer, group.group_rank, max_payload_bytes)
        if rng_state.get("bit_generator") != type(rng.bit_generator).__name__:
            raise ValueError("Checkpoint RNG bit generator differs from the current rank")
        trial_rng = type(rng.bit_generator)()
        trial_rng.state = rng_state
        restored = (state, rng_state, counters)
        return {"step": state["step"], "accumulations": state["accumulations"]}
    positions = _collective(group, "checkpoint all-rank state validation", validate_local)
    if any(item != positions[0] for item in positions[1:]):
        raise ShardedCheckpointError("Checkpoint ranks disagree about optimizer step or accumulation position")
    state, rng_state, counters = restored
    # Every potentially failing schema/value check has completed on every rank.
    # A process kill during these in-memory copies still aborts the job; it does
    # not change the committed generation and restart repeats this operation.
    optimizer.load_state_dict(state)
    rng.bit_generator.state = rng_state
    _owners(optimizer)[str(directory)] = manifest["run_id"]
    group.barrier()
    return counters
