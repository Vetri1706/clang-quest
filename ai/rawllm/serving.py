"""Bounded, authenticated, loopback-only HTTP inference using the standard library.

Connections use one request and one response. A fixed worker pool and bounded
admission semaphore cap both executing and queued connections. Deadlines and
disconnects are checked between model decode calls; an individual NumPy kernel
cannot be preempted safely by a Python thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import select
import socket
import socketserver
import stat
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from .cache import PagedLatentCache


class HTTPFailure(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code
        super().__init__(code)


class ClientDisconnected(Exception):
    """The peer closed its connection before the response was ready."""


@dataclass(frozen=True)
class ServingLimits:
    workers: int = 2
    queue_size: int = 2
    max_request_bytes: int = 16384
    max_header_bytes: int = 8192
    max_prompt_bytes: int = 8192
    max_prompt_tokens: int = 128
    max_new_tokens: int = 64
    max_context_tokens: int = 256
    max_cache_bytes: int = 32 * 1024**2
    page_size: int = 8
    io_timeout: float = 3.0
    request_timeout: float = 30.0

    def __post_init__(self):
        for name in ("workers", "max_request_bytes", "max_header_bytes", "max_prompt_bytes",
                     "max_prompt_tokens", "max_new_tokens", "max_context_tokens", "max_cache_bytes", "page_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.queue_size) is not int or self.queue_size < 0:
            raise ValueError("queue_size must be a nonnegative integer")
        if self.workers > 32 or self.queue_size > 256:
            raise ValueError("Worker/queue limits exceed this local service's supported bounds")
        if self.max_header_bytes < 128 or self.max_request_bytes > 1024**2:
            raise ValueError("Use at least 128 header bytes and at most 1 MiB request bodies")
        if any(not math.isfinite(v) or v <= 0 for v in (self.io_timeout, self.request_timeout)):
            raise ValueError("Timeouts must be finite and positive")


def _validate_token(token: str) -> str:
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9._~-]{32,512}", token):
        raise ValueError("Bearer token must be 32..512 URL-safe ASCII characters")
    return token


def load_bearer_token(token_file: str | Path, environment: str = "RAWLLM_API_TOKEN") -> tuple[str, str]:
    """Read a secret from the environment/private file, or atomically create one.

    Tokens are never printed. Existing files must be owned by the current user,
    regular, nonsymlinked, and inaccessible to group/other users. The caller is
    responsible for choosing a trusted local parent directory.
    """
    supplied = os.environ.get(environment)
    if supplied is not None:
        return _validate_token(supplied), "environment"
    path = Path(token_file)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                                 | getattr(os, "O_CLOEXEC", 0), 0o600)
        except FileExistsError:
            return load_bearer_token(path, environment)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return token, "generated_file"
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or metadata.st_uid != os.getuid():
            raise ValueError("Token file must be an owned private regular file")
        data = stream.read(514)
        if len(data) > 513:
            raise ValueError("Token file is too large")
    return _validate_token(data.decode("ascii").strip()), "file"


class InferenceService:
    def __init__(self, model, tokenizer, bearer_token: str, limits: ServingLimits | None = None):
        self.model, self.tokenizer = model, tokenizer
        self._token = _validate_token(bearer_token)
        self.limits = limits or ServingLimits()
        if model.config.vocab_size != tokenizer.vocab_size or model.precision != "fp32":
            raise ValueError("Serving requires matching tokenizer/model vocabulary and fp32 decoding")
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._metrics = {"accepted_connections": 0, "rejected_connections": 0,
                         "active_connections": 0, "queued_connections": 0,
                         "completed_generations": 0, "failed_requests": 0,
                         "cancelled_requests": 0, "generated_tokens": 0,
                         "cache_bytes": 0, "peak_cache_bytes": 0}

    def increment(self, **changes):
        with self._lock:
            for name, value in changes.items():
                self._metrics[name] += value

    def metrics(self):
        with self._lock:
            return dict(self._metrics)

    def authorized(self, header):
        if not isinstance(header, str) or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].encode("utf-8"), self._token.encode("ascii"))

    def _claim_cache(self, required):
        with self._lock:
            if self._metrics["cache_bytes"] + required > self.limits.max_cache_bytes:
                raise HTTPFailure(503, "cache_capacity")
            self._metrics["cache_bytes"] += required
            self._metrics["peak_cache_bytes"] = max(self._metrics["peak_cache_bytes"], self._metrics["cache_bytes"])

    def generate(self, payload, check_cancel):
        if not isinstance(payload, dict) or set(payload) - {"prompt", "max_tokens", "temperature", "seed"}:
            raise HTTPFailure(400, "invalid_fields")
        prompt = payload.get("prompt")
        maximum = payload.get("max_tokens", min(32, self.limits.max_new_tokens))
        temperature, seed = payload.get("temperature", 0.8), payload.get("seed", 5)
        if not isinstance(prompt, str):
            raise HTTPFailure(400, "prompt_required")
        try:
            prompt_bytes = len(prompt.encode("utf-8"))
        except UnicodeEncodeError:
            raise HTTPFailure(400, "invalid_unicode") from None
        if prompt_bytes > self.limits.max_prompt_bytes:
            raise HTTPFailure(413, "prompt_too_large")
        if type(maximum) is not int or not 0 <= maximum <= self.limits.max_new_tokens:
            raise HTTPFailure(400, "invalid_max_tokens")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise HTTPFailure(400, "invalid_seed")
        if type(temperature) not in (float, int) or not 0 <= temperature <= 10 or not math.isfinite(temperature):
            raise HTTPFailure(400, "invalid_temperature")
        check_cancel()
        ids = self.tokenizer.encode(prompt, add_bos=True)
        check_cancel()
        if len(ids) > self.limits.max_prompt_tokens:
            raise HTTPFailure(413, "prompt_token_limit")
        context = min(self.limits.max_context_tokens, self.model.config.max_seq_len)
        if len(ids) + maximum > context:
            raise HTTPFailure(400, "context_limit")
        pages = max(1, (len(ids) + maximum + self.limits.page_size - 1) // self.limits.page_size)
        config = self.model.config
        required = config.layers * pages * self.limits.page_size * (config.kv_rank + config.rope_dim) * 4 + pages * 8
        self._claim_cache(required)
        cache = None
        try:
            cache = PagedLatentCache(config, page_size=self.limits.page_size, max_pages=pages,
                                     dtype=np.float32, max_bytes=required)
            rng = np.random.default_rng(seed)
            logits = None
            for token in ids:
                check_cancel()
                logits = self.model.decode(token, cache)
            generated, reason = [], "length"
            for index in range(maximum):
                check_cancel()
                scores = np.array(logits, dtype=np.float64, copy=True)
                if not np.isfinite(scores).all():
                    raise HTTPFailure(500, "nonfinite_model_output")
                scores[self.tokenizer.pad_id] = scores[self.tokenizer.bos_id] = -np.inf
                if temperature == 0:
                    token = int(scores.argmax())
                else:
                    with np.errstate(over="ignore", under="ignore"):
                        probabilities = np.exp((scores - scores.max()) / temperature)
                    probabilities /= probabilities.sum()
                    token = int(rng.choice(len(probabilities), p=probabilities))
                if token == self.tokenizer.eos_id:
                    reason = "eos"
                    break
                generated.append(token)
                if index + 1 < maximum:
                    logits = self.model.decode(token, cache)
            check_cancel()
            result = {"generated_text": self.tokenizer.decode(generated), "token_ids": generated,
                      "prompt_tokens": len(ids), "generated_tokens": len(generated),
                      "finish_reason": reason, "seed": seed,
                      "model_status": "unvalidated_local_model"}
            self.increment(completed_generations=1, generated_tokens=len(generated))
            return result
        finally:
            if cache is not None:
                cache.release("default")
            self.increment(cache_bytes=-required)


def _response_bytes(status, payload):
    body = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    reasons = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
               405: "Method Not Allowed", 408: "Request Timeout", 411: "Length Required",
               413: "Content Too Large", 415: "Unsupported Media Type", 417: "Expectation Failed",
               431: "Request Header Fields Too Large", 500: "Internal Server Error", 503: "Service Unavailable"}
    headers = [f"HTTP/1.1 {status} {reasons.get(status, 'Error')}", "Content-Type: application/json",
               f"Content-Length: {len(body)}", "Connection: close", "Cache-Control: no-store",
               "X-Content-Type-Options: nosniff"]
    if status == 401:
        headers.append('WWW-Authenticate: Bearer realm="rawllm"')
    if status == 503:
        headers.append("Retry-After: 1")
    return ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body


class _Handler(socketserver.StreamRequestHandler):
    def setup(self):
        self.request.settimeout(self.server.service.limits.io_timeout)
        super().setup()
        self.deadline = self.server.deadlines[self.request] + self.server.service.limits.request_timeout

    def check_cancel(self):
        if self.server.service.stop_event.is_set():
            raise HTTPFailure(503, "shutting_down")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPFailure(408, "request_deadline")
        self.connection.settimeout(min(remaining, self.server.service.limits.io_timeout))
        ready, _, _ = select.select([self.connection], [], [], 0)
        if ready:
            try:
                if self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b"":
                    raise ClientDisconnected()
            except BlockingIOError:
                pass

    def _line(self, budget):
        self.check_cancel()
        line = self.rfile.readline(min(budget + 1, 4097))
        if not line:
            raise ClientDisconnected()
        if len(line) > budget or len(line) > 4096:
            raise HTTPFailure(431, "headers_too_large")
        if not line.endswith(b"\r\n"):
            raise HTTPFailure(400, "invalid_http_line")
        return line

    def _parse(self):
        budget = self.server.service.limits.max_header_bytes
        first = self._line(budget)
        budget -= len(first)
        try:
            method, target, version = first[:-2].decode("ascii").split(" ")
        except (ValueError, UnicodeDecodeError):
            raise HTTPFailure(400, "invalid_request_line") from None
        if version not in ("HTTP/1.0", "HTTP/1.1") or target not in ("/health", "/metrics", "/generate"):
            raise HTTPFailure(404 if target.startswith("/") else 400, "invalid_route")
        headers = {}
        for _ in range(33):
            line = self._line(budget)
            budget -= len(line)
            if line == b"\r\n":
                break
            try:
                key, value = line[:-2].decode("ascii").split(":", 1)
            except (ValueError, UnicodeDecodeError):
                raise HTTPFailure(400, "invalid_header") from None
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise HTTPFailure(400, "invalid_header")
            key = key.lower()
            if key in headers:
                raise HTTPFailure(400, "duplicate_header")
            headers[key] = value.strip()
        else:
            raise HTTPFailure(431, "too_many_headers")
        try:
            host = urlsplit("//" + headers.get("host", ""))
            allowed = {self.server.server_address[0], "127.0.0.1", "localhost", "::1"}
            if host.hostname not in allowed or host.username is not None or host.password is not None or host.path or host.query or host.fragment:
                raise ValueError("host")
            if host.port is not None and host.port != self.server.server_address[1]:
                raise ValueError("port")
        except ValueError:
            raise HTTPFailure(403, "invalid_host") from None
        if "transfer-encoding" in headers:
            raise HTTPFailure(400, "transfer_encoding_unsupported")
        if "expect" in headers:
            raise HTTPFailure(417, "expectation_unsupported")
        if method not in ("GET", "POST"):
            raise HTTPFailure(405, "method_not_allowed")
        if method == "GET":
            if target == "/generate" or headers.get("content-length", "0") != "0":
                raise HTTPFailure(400, "invalid_get")
            return target, None
        if target != "/generate":
            raise HTTPFailure(405, "method_not_allowed")
        if not self.server.service.authorized(headers.get("authorization")):
            raise HTTPFailure(401, "unauthorized")
        length = headers.get("content-length")
        if length is None:
            raise HTTPFailure(411, "content_length_required")
        if not re.fullmatch(r"[0-9]{1,10}", length):
            raise HTTPFailure(400, "invalid_content_length")
        length = int(length)
        if length > self.server.service.limits.max_request_bytes:
            raise HTTPFailure(413, "request_too_large")
        if headers.get("content-type", "").lower().split(";", 1)[0].strip() != "application/json":
            raise HTTPFailure(415, "json_required")
        body = bytearray()
        while len(body) < length:
            self.check_cancel()
            part = self.rfile.read1(min(4096, length - len(body)))
            if not part:
                raise HTTPFailure(400, "truncated_body")
            body.extend(part)
        def reject_constant(_):
            raise ValueError("nonfinite")
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate json field")
                result[key] = value
            return result
        try:
            payload = json.loads(body.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_fields)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise HTTPFailure(400, "invalid_json") from None
        return target, payload

    def handle(self):
        try:
            target, payload = self._parse()
            if target == "/health":
                result = {"status": "ready", "model_status": "unvalidated_local_model"}
            elif target == "/metrics":
                result = self.server.service.metrics()
            else:
                result = self.server.service.generate(payload, self.check_cancel)
            self.check_cancel()
            response = _response_bytes(200, result)
        except ClientDisconnected:
            self.server.service.increment(cancelled_requests=1)
            return
        except (TimeoutError, socket.timeout):
            self.server.service.increment(failed_requests=1)
            response = _response_bytes(408, {"error": "request_timeout"})
        except HTTPFailure as error:
            self.server.service.increment(failed_requests=1)
            response = _response_bytes(error.status, {"error": error.code})
        except Exception:
            self.server.service.increment(failed_requests=1)
            response = _response_bytes(500, {"error": "internal_error"})
        try:
            self.connection.settimeout(self.server.service.limits.io_timeout)
            self.connection.sendall(response)
        except (OSError, TimeoutError):
            self.server.service.increment(cancelled_requests=1)


class BoundedInferenceServer(socketserver.TCPServer):
    """One accept loop, a fixed worker pool, and finite connection admission."""

    allow_reuse_address = True

    def __init__(self, address, service: InferenceService):
        host, port = address
        if host == "localhost":
            host = "127.0.0.1"
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError("Serve only an explicit loopback IP or localhost") from None
        if not ip.is_loopback:
            raise ValueError("This local inference service cannot bind a non-loopback address")
        self.address_family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        self.service = service
        self.request_queue_size = service.limits.workers + service.limits.queue_size
        self._admission = threading.BoundedSemaphore(self.request_queue_size)
        self._pending = queue.Queue(maxsize=self.request_queue_size)
        self._active_lock = threading.Lock()
        self._active_sockets = set()
        self.deadlines = {}
        self._closed = False
        self._workers = []
        super().__init__((str(ip), port), _Handler)
        self._workers = [threading.Thread(target=self._worker, name=f"rawllm-http-{i}", daemon=False)
                         for i in range(service.limits.workers)]
        for worker in self._workers:
            worker.start()

    def process_request(self, request, client_address):
        if self.service.stop_event.is_set() or not self._admission.acquire(blocking=False):
            self.service.increment(rejected_connections=1)
            try:
                request.settimeout(min(0.1, self.service.limits.io_timeout))
                request.sendall(_response_bytes(503, {"error": "server_busy"}))
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        self.deadlines[request] = time.monotonic()
        self.service.increment(accepted_connections=1, queued_connections=1)
        self._pending.put_nowait((request, client_address))

    def _worker(self):
        while True:
            item = self._pending.get()
            try:
                if item is None:
                    return
                request, client_address = item
                self.service.increment(queued_connections=-1, active_connections=1)
                with self._active_lock:
                    self._active_sockets.add(request)
                try:
                    self.finish_request(request, client_address)
                except Exception:
                    self.service.increment(failed_requests=1)
                finally:
                    self.shutdown_request(request)
                    with self._active_lock:
                        self._active_sockets.discard(request)
                    self.deadlines.pop(request, None)
                    self.service.increment(active_connections=-1)
                    self._admission.release()
            finally:
                self._pending.task_done()

    def server_close(self):
        if self._closed:
            return
        self._closed = True
        self.service.stop_event.set()
        super().server_close()
        with self._active_lock:
            for connection in tuple(self._active_sockets):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        for _ in self._workers:
            self._pending.put(None)
        for worker in self._workers:
            worker.join()


def server_from_run(run, bearer_token, address=("127.0.0.1", 8080), limits=None):
    """Load the same checksummed local run format as generate.py."""
    from generate import load_run
    model, tokenizer = load_run(run)
    return BoundedInferenceServer(address, InferenceService(model, tokenizer, bearer_token, limits))
