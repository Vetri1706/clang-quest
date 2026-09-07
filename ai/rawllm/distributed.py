"""Authenticated TCP reference collectives and a versioned parameter server.

The coordinator routes host-memory NumPy buffers. This implements collective
semantics, not ring collectives, RDMA, GPU direct communication, or fault-tolerant
membership. A failed or timed-out participant aborts the run. HMAC authenticates
and integrity-protects frames; it does not encrypt their contents.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import secrets
import socket
import struct
import threading
import time
from typing import Any

import numpy as np


class DistributedError(RuntimeError):
    """The communication session cannot complete the requested operation."""


class ProtocolError(DistributedError):
    """An invalid or unauthenticated wire message was received."""


_HEADER = struct.Struct("!IQ")
_MAX_METADATA = 65536
_DTYPES = {"<f2", "<f4", "<f8", "<i2", "<i4", "<i8", "|i1", "|u1", "<u2", "<u4", "<u8", "|b1"}


def _exact(sock: socket.socket, size: int) -> bytes:
    pieces = bytearray()
    while len(pieces) < size:
        try:
            chunk = sock.recv(min(size - len(pieces), 1024 * 1024))
        except (TimeoutError, OSError) as error:
            raise DistributedError(f"socket receive failed: {error}") from error
        if not chunk:
            raise DistributedError("peer closed connection")
        pieces.extend(chunk)
    return bytes(pieces)


def _send_frame(sock: socket.socket, token: bytes, seq: int, message: dict,
                arrays: list[np.ndarray], max_payload: int) -> None:
    if len(arrays) > 1024:
        raise ProtocolError("too many arrays in one frame")
    descriptors, buffers, offset = [], [], 0
    for value in arrays:
        array = np.asarray(value)
        dtype = array.dtype.newbyteorder("<")
        if dtype.str not in _DTYPES or array.ndim > 16:
            raise ProtocolError("unsupported ndarray dtype or rank")
        if offset + array.nbytes > max_payload:
            raise ProtocolError("payload exceeds configured bound")
        contiguous = np.ascontiguousarray(array, dtype=dtype)
        descriptors.append({"dtype": dtype.str, "shape": list(array.shape), "offset": offset, "nbytes": array.nbytes})
        buffers.append(contiguous.tobytes())
        offset += array.nbytes
    envelope = {"seq": seq, "message": message, "arrays": descriptors}
    try:
        metadata = json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode("utf8")
    except (TypeError, ValueError) as error:
        raise ProtocolError("metadata must contain finite JSON values") from error
    if len(metadata) > _MAX_METADATA:
        raise ProtocolError("metadata exceeds configured bound")
    payload = b"".join(buffers)
    header = _HEADER.pack(len(metadata), len(payload))
    digest = hmac.new(token, header + metadata + payload, hashlib.sha256).digest()
    try:
        sock.sendall(header + metadata + payload + digest)
    except (TimeoutError, OSError) as error:
        raise DistributedError(f"socket send failed: {error}") from error


def _recv_frame(sock: socket.socket, token: bytes, expected_seq: int,
                max_payload: int) -> tuple[dict, list[np.ndarray]]:
    header = _exact(sock, _HEADER.size)
    metadata_size, payload_size = _HEADER.unpack(header)
    if metadata_size > _MAX_METADATA or payload_size > max_payload:
        raise ProtocolError("incoming frame exceeds configured bound")
    metadata, payload = _exact(sock, metadata_size), _exact(sock, payload_size)
    received_mac = _exact(sock, 32)
    expected_mac = hmac.new(token, header + metadata + payload, hashlib.sha256).digest()
    if not hmac.compare_digest(received_mac, expected_mac):
        raise ProtocolError("frame authentication failed")
    try:
        envelope = json.loads(metadata)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("malformed JSON metadata") from error
    if not isinstance(envelope, dict) or type(envelope.get("seq")) is not int or envelope.get("seq") != expected_seq:
        raise ProtocolError("frame sequence mismatch")
    descriptors = envelope.get("arrays")
    message = envelope.get("message")
    if not isinstance(message, dict) or not isinstance(descriptors, list) or len(descriptors) > 1024:
        raise ProtocolError("invalid frame envelope")
    arrays, offset = [], 0
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ProtocolError("invalid ndarray descriptor")
        dtype_text, shape = descriptor.get("dtype"), descriptor.get("shape")
        if not isinstance(dtype_text, str) or dtype_text not in _DTYPES:
            raise ProtocolError("unsupported ndarray dtype")
        if not isinstance(shape, list) or len(shape) > 16 or any(type(d) is not int or d < 0 or d > 2**31 for d in shape):
            raise ProtocolError("invalid ndarray shape")
        dtype = np.dtype(dtype_text)
        nbytes = math.prod(shape) * dtype.itemsize
        if (type(descriptor.get("offset")) is not int or type(descriptor.get("nbytes")) is not int
                or descriptor.get("offset") != offset or descriptor.get("nbytes") != nbytes
                or offset + nbytes > len(payload)):
            raise ProtocolError("invalid ndarray byte extent")
        arrays.append(np.frombuffer(payload, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy())
        offset += nbytes
    if offset != len(payload):
        raise ProtocolError("unclaimed payload bytes")
    return message, arrays


@dataclass
class _Rendezvous:
    op: str
    arguments: dict
    submissions: dict = field(default_factory=dict)
    results: dict | None = None
    consumed: int = 0
    error: str | None = None


class CollectiveServer:
    """Threaded, bounded coordinator. Use a new random token for every run.

    A world-size set of ranks must connect before the first collective deadline.
    All requests have deadlines. Shutdown closes every accepted socket and joins
    handler threads. Host defaults to loopback; remote operation requires a
    private encrypted tunnel because the HMAC transport provides no secrecy.
    """

    def __init__(self, world_size: int, host: str = "127.0.0.1", port: int = 0,
                 token: str | None = None, timeout: float = 10.0,
                 max_payload_bytes: int = 64 * 1024 * 1024,
                 max_queued_bytes: int = 256 * 1024 * 1024):
        if world_size < 1 or timeout <= 0 or max_payload_bytes <= 0 or max_queued_bytes < max_payload_bytes:
            raise ValueError("invalid server configuration")
        self.world_size, self.host, self.port = world_size, host, port
        self.token = token or secrets.token_hex(32)
        if len(self.token.encode()) < 32:
            raise ValueError("shared token must contain at least 32 bytes")
        self.timeout, self.max_payload = timeout, max_payload_bytes
        self.max_queued_bytes = max_queued_bytes
        self._key = self.token.encode()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._handlers: list[threading.Thread] = []
        self._sockets: set[socket.socket] = set()
        self._connected: set[int] = set()
        self._closed: set[int] = set()
        self._abort_reason: str | None = None
        self._collectives: dict[tuple, _Rendezvous] = {}
        self._mail = defaultdict(deque)
        self._queued_bytes = 0
        self._parameters: dict[str, dict] = {}
        self._collective_bytes = 0
        self._parameter_pending_bytes = 0

    def start(self) -> "CollectiveServer":
        if self._listener is not None:
            raise DistributedError("server is already started")
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, self.port))
        self.port = self._listener.getsockname()[1]
        self._listener.listen(self.world_size)
        self._listener.settimeout(0.2)
        self._accept_thread = threading.Thread(target=self._accept, name="rawllm-coordinator", daemon=True)
        self._accept_thread.start()
        return self

    def __enter__(self) -> "CollectiveServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def abort(self, reason: str) -> None:
        with self._condition:
            if self._abort_reason is None:
                self._abort_reason = reason
            self._condition.notify_all()

    def close(self) -> None:
        self._stop.set()
        self.abort("server shut down")
        if self._listener is not None:
            self._listener.close()
        with self._condition:
            sockets = list(self._sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1)
        for handler in self._handlers:
            handler.join(timeout=1)

    def _accept(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                sock, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            sock.settimeout(self.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._condition:
                if len(self._sockets) >= self.world_size:
                    sock.close()
                    continue
                self._sockets.add(sock)
            handler = threading.Thread(target=self._serve, args=(sock,), daemon=True)
            self._handlers.append(handler)
            handler.start()

    def _serve(self, sock: socket.socket) -> None:
        rank, outgoing, incoming, orderly = None, 0, 0, False
        try:
            hello, arrays = _recv_frame(sock, self._key, incoming, self.max_payload)
            incoming += 1
            rank = hello.get("rank")
            if hello.get("op") != "hello" or arrays or type(rank) is not int or not 0 <= rank < self.world_size or hello.get("world_size") != self.world_size:
                raise ProtocolError("invalid rank handshake")
            with self._condition:
                if rank in self._connected:
                    raise ProtocolError("duplicate rank handshake")
                self._connected.add(rank)
            _send_frame(sock, self._key, outgoing, {"ok": True}, [], self.max_payload)
            outgoing += 1
            while not self._stop.is_set():
                request, arrays = _recv_frame(sock, self._key, incoming, self.max_payload)
                incoming += 1
                if request.get("op") == "close":
                    with self._condition:
                        self._closed.add(rank)
                        self._condition.notify_all()
                    _send_frame(sock, self._key, outgoing, {"ok": True}, [], self.max_payload)
                    orderly = True
                    break
                response, result = self._dispatch(rank, request, arrays)
                _send_frame(sock, self._key, outgoing, {"ok": True, **response}, result, self.max_payload)
                outgoing += 1
        except Exception as error:
            if not self._stop.is_set():
                try:
                    _send_frame(sock, self._key, outgoing, {"ok": False, "error": str(error)}, [], self.max_payload)
                except Exception:
                    pass
                self.abort(f"rank {rank}: {type(error).__name__}: {error}")
        finally:
            if rank is not None and not orderly and not self._stop.is_set():
                self.abort(f"rank {rank} disconnected unexpectedly")
            with self._condition:
                self._sockets.discard(sock)
            sock.close()

    def _wait(self, predicate, deadline: float) -> None:
        while not predicate():
            if self._abort_reason is not None:
                raise DistributedError(self._abort_reason)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DistributedError("distributed operation timed out waiting for participants")
            self._condition.wait(min(remaining, 0.2))

    def _members(self, request: dict, rank: int) -> tuple[int, ...]:
        members = request.get("members")
        if not isinstance(members, list) or not members or any(type(m) is not int or not 0 <= m < self.world_size for m in members):
            raise ProtocolError("invalid group members")
        if members != sorted(set(members)) or rank not in members:
            raise ProtocolError("group must be sorted, unique, and contain caller")
        return tuple(members)

    def _dispatch(self, rank: int, request: dict, arrays: list[np.ndarray]) -> tuple[dict, list[np.ndarray]]:
        deadline = time.monotonic() + self.timeout
        op = request.get("op")
        with self._condition:
            if self._abort_reason:
                raise DistributedError(self._abort_reason)
            if op in {"send", "recv"}:
                peer, tag = request.get("peer"), request.get("tag")
                if type(peer) is not int or not 0 <= peer < self.world_size or not isinstance(tag, str) or len(tag) > 256:
                    raise ProtocolError("invalid point-to-point peer or tag")
                if op == "send":
                    if len(arrays) != 1:
                        raise ProtocolError("send requires exactly one ndarray")
                    if self._queued_bytes + arrays[0].nbytes > self.max_queued_bytes or sum(map(len, self._mail.values())) >= 4096:
                        raise DistributedError("point-to-point queue bound exceeded")
                    self._mail[(rank, peer, tag)].append(arrays[0])
                    self._queued_bytes += arrays[0].nbytes
                    self._condition.notify_all()
                    return {}, []
                if arrays:
                    raise ProtocolError("recv must not include arrays")
                key = (peer, rank, tag)
                self._wait(lambda: bool(self._mail[key]) or peer in self._closed, deadline)
                if not self._mail[key]:
                    raise DistributedError("sender closed without matching send")
                result = self._mail[key].popleft()
                self._queued_bytes -= result.nbytes
                if not self._mail[key]:
                    del self._mail[key]
                return {}, [result]
            if op in {"parameter_init", "parameter_pull", "parameter_push"}:
                return self._parameter(rank, request, arrays, deadline)
            supported = {"all_reduce", "all_gather", "scatter", "gather", "reduce_scatter", "broadcast", "barrier"}
            if op not in supported:
                raise ProtocolError(f"unsupported operation {op!r}")
            members = self._members(request, rank)
            counter = request.get("counter")
            if type(counter) is not int or counter < 0:
                raise ProtocolError("invalid collective counter")
            arguments = {"root": request.get("root"), "reduction": request.get("reduction")}
            key = (members, counter)
            if key not in self._collectives:
                if len(self._collectives) >= 1024:
                    raise DistributedError("too many outstanding collective rendezvous")
                self._collectives[key] = _Rendezvous(op, arguments)
            state = self._collectives[key]
            if state.op != op or state.arguments != arguments or rank in state.submissions:
                raise ProtocolError("collective order or arguments disagree across ranks")
            submission_bytes = sum(array.nbytes for array in arrays)
            if self._collective_bytes + submission_bytes > self.max_queued_bytes:
                raise DistributedError("outstanding collective payload bound exceeded")
            state.submissions[rank] = arrays
            self._collective_bytes += submission_bytes
            if len(state.submissions) == len(members):
                try:
                    state.results = self._calculate(op, arguments, members, state.submissions)
                except Exception as error:
                    state.error = str(error)
                self._condition.notify_all()
            self._wait(lambda: state.results is not None or state.error is not None, deadline)
            if state.error:
                raise DistributedError(state.error)
            assert state.results is not None
            result = state.results[rank]
            state.consumed += 1
            if state.consumed == len(members):
                self._collective_bytes -= sum(array.nbytes for submission in state.submissions.values() for array in submission)
                del self._collectives[key]
            return {}, result

    @staticmethod
    def _calculate(op: str, arguments: dict, members: tuple, submissions: dict) -> dict:
        root = arguments["root"]
        reduction = arguments["reduction"]
        if op in {"gather", "scatter", "broadcast"} and root not in members:
            raise ProtocolError("root is outside collective group")
        if op == "barrier":
            if any(submissions[r] for r in members):
                raise ProtocolError("barrier cannot carry arrays")
            return {r: [] for r in members}
        if op in {"scatter", "broadcast"}:
            count = len(members) if op == "scatter" else 1
            if len(submissions[root]) != count or any(submissions[r] for r in members if r != root):
                raise ProtocolError("invalid root/nonroot payload for scatter or broadcast")
            return {r: [submissions[root][i if op == "scatter" else 0]] for i, r in enumerate(members)}
        if any(len(submissions[r]) != 1 for r in members):
            raise ProtocolError("collective requires one array per rank")
        values = [submissions[r][0] for r in members]
        if op == "all_gather":
            return {r: values for r in members}
        if op == "gather":
            return {r: values if r == root else [] for r in members}
        first = values[0]
        if first.dtype.kind not in "fi" or any(v.shape != first.shape or v.dtype != first.dtype for v in values):
            raise ProtocolError("reduction requires matching numeric shapes and dtypes")
        if reduction not in {"sum", "mean"}:
            raise ProtocolError("reduction must be sum or mean")
        if reduction == "mean" and first.dtype.kind != "f":
            raise ProtocolError("mean reduction requires a floating-point dtype")
        result = first.copy()
        for value in values[1:]:
            np.add(result, value, out=result)
        if reduction == "mean":
            result /= len(members)
        if op == "all_reduce":
            return {r: [result] for r in members}
        if first.ndim != 1:
            raise ProtocolError("reduce_scatter requires a flat input")
        pieces = np.array_split(result, len(members))
        return {r: [pieces[i]] for i, r in enumerate(members)}

    def _parameter(self, rank: int, request: dict, arrays: list[np.ndarray], deadline: float) -> tuple[dict, list[np.ndarray]]:
        name, op = request.get("name"), request["op"]
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ProtocolError("invalid parameter name")
        if op == "parameter_init":
            if rank != 0 or name in self._parameters or len(arrays) != 1 or arrays[0].dtype != np.float32:
                raise ProtocolError("only rank zero may initialize a new FP32 parameter")
            if len(self._parameters) >= 1024:
                raise ProtocolError("parameter server name-count bound exceeded")
            if sum(p["value"].nbytes for p in self._parameters.values()) + arrays[0].nbytes > self.max_queued_bytes:
                raise ProtocolError("parameter server storage bound exceeded")
            workers = request.get("workers")
            if not isinstance(workers, list) or not workers or workers != sorted(set(workers)) or any(type(r) is not int or not 0 <= r < self.world_size for r in workers):
                raise ProtocolError("invalid parameter worker set")
            if not np.isfinite(arrays[0]).all():
                raise ProtocolError("parameters must be finite")
            self._parameters[name] = {"value": arrays[0], "version": 0, "workers": tuple(workers), "pending": {}, "lr": None}
            self._condition.notify_all()
            return {"version": 0}, []
        if name not in self._parameters:
            raise ProtocolError("parameter was not initialized")
        parameter = self._parameters[name]
        if op == "parameter_pull":
            if arrays:
                raise ProtocolError("pull cannot carry arrays")
            return {"version": parameter["version"]}, [parameter["value"].copy()]
        version, lr = request.get("version"), request.get("learning_rate")
        if version != parameter["version"]:
            return {"accepted": False, "version": parameter["version"], "reason": "stale_version"}, []
        if rank not in parameter["workers"] or rank in parameter["pending"]:
            raise ProtocolError("unauthorized or duplicate parameter contribution")
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr < 0:
            raise ProtocolError("invalid learning rate")
        if len(arrays) != 1 or arrays[0].shape != parameter["value"].shape or arrays[0].dtype != np.float32 or not np.isfinite(arrays[0]).all():
            raise ProtocolError("gradient must be finite FP32 with matching shape")
        if parameter["lr"] is not None and parameter["lr"] != lr:
            raise ProtocolError("workers disagree about the learning rate")
        if self._parameter_pending_bytes + arrays[0].nbytes > self.max_queued_bytes:
            raise ProtocolError("pending parameter gradient bound exceeded")
        parameter["lr"] = lr
        parameter["pending"][rank] = arrays[0]
        self._parameter_pending_bytes += arrays[0].nbytes
        if len(parameter["pending"]) == len(parameter["workers"]):
            gradient = np.zeros_like(parameter["value"])
            for worker in parameter["workers"]:
                gradient += parameter["pending"][worker]
            gradient /= len(parameter["workers"])
            candidate = parameter["value"] - lr * gradient
            if not np.isfinite(candidate).all():
                raise DistributedError("parameter update overflowed; transaction aborted")
            parameter["value"] = candidate
            parameter["version"] += 1
            self._parameter_pending_bytes -= sum(value.nbytes for value in parameter["pending"].values())
            parameter["pending"] = {}
            parameter["lr"] = None
            self._condition.notify_all()
        else:
            self._wait(lambda: parameter["version"] != version, deadline)
        return {"accepted": True, "version": parameter["version"]}, [parameter["value"].copy()]


class ProcessGroup:
    """One ordered request stream per rank, shared by subgroup views.

    Root and peer arguments are GLOBAL ranks. Within each sorted subgroup,
    collective calls must occur in the same order on all participating ranks.
    Calls on one connection are serialized; use send-then-recv ordering to avoid
    attempting simultaneous blocking receives on one rank.
    """

    def __init__(self, rank: int, world_size: int, host: str, port: int, token: str,
                 timeout: float = 10.0, max_payload_bytes: int = 64 * 1024 * 1024):
        if not 0 <= rank < world_size or timeout <= 0 or len(token.encode()) < 32:
            raise ValueError("invalid process group configuration")
        self.rank, self.world_size = rank, world_size
        self.members = tuple(range(world_size))
        self.group_rank, self.size = rank, world_size
        self._owner = self
        self._counters = defaultdict(int)
        self._send_seq = self._recv_seq = 0
        self._max_payload = max_payload_bytes
        self._key = token.encode()
        self._lock = threading.Lock()
        self._closed = False
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._socket.settimeout(timeout + 0.5)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._rpc({"op": "hello", "rank": rank, "world_size": world_size}, [])
        except Exception:
            self._socket.close()
            raise

    def __enter__(self) -> "ProcessGroup":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def subgroup(self, members: list[int] | tuple[int, ...]) -> "ProcessGroup":
        members = tuple(sorted(members))
        if not members or len(set(members)) != len(members) or self.rank not in members or any(m not in self.members for m in members):
            raise ValueError("subgroup must contain caller and distinct parent ranks")
        group = object.__new__(ProcessGroup)
        group.rank, group.world_size, group.members = self.rank, self.world_size, members
        group.group_rank, group.size = members.index(self.rank), len(members)
        group._owner = self._owner
        return group

    def _rpc(self, message: dict, arrays: list[np.ndarray]) -> tuple[dict, list[np.ndarray]]:
        owner = self._owner
        with owner._lock:
            if owner._closed:
                raise DistributedError("process group is closed")
            _send_frame(owner._socket, owner._key, owner._send_seq, message, arrays, owner._max_payload)
            owner._send_seq += 1
            response, result = _recv_frame(owner._socket, owner._key, owner._recv_seq, owner._max_payload)
            owner._recv_seq += 1
            if not response.get("ok"):
                raise DistributedError(response.get("error", "remote operation failed"))
            return response, result

    def _collective(self, op: str, arrays: list[np.ndarray], root: int | None = None,
                    reduction: str | None = None) -> list[np.ndarray]:
        counter = self._owner._counters[self.members]
        self._owner._counters[self.members] += 1
        return self._rpc({"op": op, "members": list(self.members), "counter": counter,
                          "root": root, "reduction": reduction}, arrays)[1]

    def all_reduce(self, value: np.ndarray, reduction: str = "sum") -> np.ndarray:
        return self._collective("all_reduce", [value], reduction=reduction)[0]

    def all_gather(self, value: np.ndarray) -> list[np.ndarray]:
        return self._collective("all_gather", [value])

    def scatter(self, values: list[np.ndarray] | None, root: int = 0) -> np.ndarray:
        return self._collective("scatter", [] if values is None else values, root=root)[0]

    def gather(self, value: np.ndarray, root: int = 0) -> list[np.ndarray] | None:
        result = self._collective("gather", [value], root=root)
        return result if self.rank == root else None

    def broadcast(self, value: np.ndarray | None, root: int = 0) -> np.ndarray:
        return self._collective("broadcast", [] if value is None else [value], root=root)[0]

    def reduce_scatter(self, value: np.ndarray, reduction: str = "sum") -> np.ndarray:
        return self._collective("reduce_scatter", [value], reduction=reduction)[0]

    def barrier(self) -> None:
        self._collective("barrier", [])

    def send(self, value: np.ndarray, dst: int, tag: str = "default") -> None:
        self._rpc({"op": "send", "peer": dst, "tag": tag}, [value])

    def recv(self, src: int, tag: str = "default") -> np.ndarray:
        return self._rpc({"op": "recv", "peer": src, "tag": tag}, [])[1][0]

    def close(self) -> None:
        if self._owner is not self:
            return
        if not self._closed:
            try:
                self._rpc({"op": "close"}, [])
            except (DistributedError, OSError):
                pass
            finally:
                self._closed = True
                self._socket.close()


class ParameterServerClient:
    """Synchronous versioned SGD aggregation, separate from sharded AdamW."""

    def __init__(self, group: ProcessGroup):
        self.group = group

    def initialize(self, name: str, value: np.ndarray, workers: list[int] | None = None) -> None:
        self.group._rpc({"op": "parameter_init", "name": name,
                         "workers": list(self.group.members) if workers is None else sorted(workers)},
                        [np.asarray(value, dtype=np.float32)])

    def pull(self, name: str) -> tuple[int, np.ndarray]:
        metadata, arrays = self.group._rpc({"op": "parameter_pull", "name": name}, [])
        return metadata["version"], arrays[0]

    def push(self, name: str, version: int, gradient: np.ndarray,
             learning_rate: float) -> tuple[bool, int, np.ndarray | None]:
        metadata, arrays = self.group._rpc({"op": "parameter_push", "name": name,
                                          "version": version, "learning_rate": learning_rate},
                                         [np.asarray(gradient, dtype=np.float32)])
        return metadata["accepted"], metadata["version"], arrays[0] if arrays else None


@dataclass(frozen=True)
class ParallelTopology:
    data: int = 2
    pipeline: int = 2
    tensor: int = 2

    def __post_init__(self) -> None:
        if min(self.data, self.pipeline, self.tensor) < 1:
            raise ValueError("parallel dimensions must be positive")

    @property
    def world_size(self) -> int:
        return self.data * self.pipeline * self.tensor

    def rank(self, data: int, pipeline: int, tensor: int) -> int:
        if not (0 <= data < self.data and 0 <= pipeline < self.pipeline and 0 <= tensor < self.tensor):
            raise ValueError("coordinate outside topology")
        return (data * self.pipeline + pipeline) * self.tensor + tensor

    def coordinates(self, rank: int) -> tuple[int, int, int]:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank outside topology")
        dp, rest = divmod(rank, self.pipeline * self.tensor)
        pp, tp = divmod(rest, self.tensor)
        return dp, pp, tp

    def groups(self, rank: int) -> dict[str, tuple[int, ...]]:
        dp, pp, tp = self.coordinates(rank)
        return {"data": tuple(self.rank(d, pp, tp) for d in range(self.data)),
                "pipeline": tuple(self.rank(dp, p, tp) for p in range(self.pipeline)),
                "tensor": tuple(self.rank(dp, pp, t) for t in range(self.tensor))}
