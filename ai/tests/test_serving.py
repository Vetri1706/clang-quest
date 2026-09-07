"""Real-socket tests for bounded admission, protocol handling, and cache cleanup."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import http.client
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from generate import generate
from rawllm.cache import PagedLatentCache
from rawllm.model import Config, Transformer
from rawllm.serving import (
    BoundedInferenceServer, InferenceService, ServingLimits, load_bearer_token,
)
from rawllm.tokenizer import ByteBPETokenizer


TOKEN = "test-only-token-" + "a" * 32


def eventually(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition did not become true before timeout")
        time.sleep(0.005)


@contextmanager
def running(model, tokenizer, **limits):
    settings = replace(ServingLimits(), **limits)
    service = InferenceService(model, tokenizer, TOKEN, settings)
    server = BoundedInferenceServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, name="test-http-accept")
    thread.start()
    try:
        yield server, service
    finally:
        service.stop_event.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive() or any(worker.is_alive() for worker in server._workers):
            raise AssertionError("HTTP service did not terminate its worker threads")


def request(server, payload=None, path="/generate", token=TOKEN, method="POST"):
    client = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    try:
        client.request(method, path, body=body, headers=headers)
        response = client.getresponse()
        raw = response.read()
        return response.status, json.loads(raw), dict(response.getheaders())
    finally:
        client.close()


def raw_request(server, body):
    with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=3) as connection:
        connection.sendall(body)
        result = bytearray()
        while True:
            try:
                part = connection.recv(4096)
            except ConnectionResetError:
                break
            if not part:
                break
            result.extend(part)
    header, payload = bytes(result).split(b"\r\n\r\n", 1)
    return int(header.split(b" ")[1]), json.loads(payload)


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteBPETokenizer()
        self.model = Transformer(Config(vocab_size=self.tokenizer.vocab_size, dim=8, layers=2,
                                       heads=2, q_rank=4, kv_rank=3, content_dim=2, rope_dim=2,
                                       value_dim=2, hidden_dim=12, max_seq_len=64), seed=53)

    def test_authenticated_generation_matches_standalone_paged_generation(self):
        expected = generate(self.model, self.tokenizer, "C++", max_tokens=6, temperature=0.8, seed=23)
        with running(self.model, self.tokenizer) as (server, service):
            status, result, headers = request(server, {"prompt": "C++", "max_tokens": 6, "temperature": 0.8, "seed": 23})
            self.assertEqual(status, 200)
            self.assertEqual(result["generated_text"], expected)
            self.assertEqual(result["prompt_tokens"], 4)
            self.assertEqual(result["generated_tokens"], len(result["token_ids"]))
            self.assertEqual(headers["Connection"], "close")
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertEqual(service.metrics()["cache_bytes"], 0)
            self.assertEqual(service.metrics()["completed_generations"], 1)

    def test_health_and_metrics_contain_no_credentials_or_paths(self):
        with running(self.model, self.tokenizer) as (server, service):
            for path in ("/health", "/metrics"):
                status, payload, _ = request(server, path=path, token=None, method="GET")
                self.assertEqual(status, 200)
                serialized = json.dumps(payload)
                self.assertNotIn(TOKEN, serialized)
                self.assertNotIn(str(Path.home()), serialized)
            status, payload, headers = request(server, {"prompt": ""}, token=None)
            self.assertEqual(status, 401)
            self.assertEqual(payload, {"error": "unauthorized"})
            self.assertIn("WWW-Authenticate", headers)
            self.assertEqual(request(server, {"prompt": ""}, token="wrong")[0], 401)

    def test_independent_request_rng_is_repeatable_during_concurrency(self):
        with running(self.model, self.tokenizer, workers=2, queue_size=2) as (server, _):
            def sample(seed):
                return request(server, {"prompt": "x", "max_tokens": 8, "seed": seed})
            with ThreadPoolExecutor(max_workers=2) as executor:
                a, b = list(executor.map(sample, [7, 11]))
            repeated = sample(7)
            self.assertEqual(a[0], 200)
            self.assertEqual(b[0], 200)
            self.assertEqual(a[1]["token_ids"], repeated[1]["token_ids"])
            self.assertNotEqual(a[1]["token_ids"], b[1]["token_ids"])

    def test_strict_framing_and_json_rejections(self):
        with running(self.model, self.tokenizer) as (server, _):
            prefix = f"POST /generate HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\nAuthorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\n".encode()
            cases = [
                (b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}", 400, "duplicate_header"),
                (b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}", 400, "transfer_encoding_unsupported"),
                (b"Content-Length: -1\r\n\r\n", 400, "invalid_content_length"),
                (b"Expect: 100-continue\r\nContent-Length: 2\r\n\r\n{}", 417, "expectation_unsupported"),
                (b"\r\n{}", 411, "content_length_required"),
                (b"Content-Length: 1\r\n\r\n{", 400, "invalid_json"),
                (b"Content-Length: 9\r\n\r\n[NaN,123]", 400, "invalid_json"),
                (b'Content-Length: 15\r\n\r\n{"x":1,"x":2}  ', 400, "invalid_json"),
            ]
            for suffix, expected_status, expected_code in cases:
                with self.subTest(suffix=suffix):
                    status, result = raw_request(server, prefix + suffix)
                    self.assertEqual(status, expected_status)
                    self.assertEqual(result, {"error": expected_code})

    def test_oversized_headers_and_body_are_rejected_before_generation(self):
        with running(self.model, self.tokenizer, max_request_bytes=128, max_header_bytes=512) as (server, service):
            host = f"Host: 127.0.0.1:{server.server_address[1]}\r\n".encode()
            status, result = raw_request(server, b"GET /health HTTP/1.1\r\n" + host + b"X-Large: " + b"a" * 600 + b"\r\n\r\n")
            self.assertEqual(status, 431)
            status, result = raw_request(server, b"POST /generate HTTP/1.1\r\n" + host +
                f"Authorization: Bearer {TOKEN}\r\nContent-Length: 129\r\nContent-Type: application/json\r\n\r\n".encode())
            self.assertEqual(status, 413)
            self.assertEqual(result["error"], "request_too_large")
            self.assertEqual(service.metrics()["peak_cache_bytes"], 0)

    def test_prompt_generation_and_context_limits(self):
        with running(self.model, self.tokenizer, max_prompt_bytes=8, max_prompt_tokens=5,
                     max_context_tokens=7, max_new_tokens=4) as (server, service):
            cases = [({"prompt": "123456789"}, 413, "prompt_too_large"),
                     ({"prompt": "12345"}, 413, "prompt_token_limit"),
                     ({"prompt": "1234", "max_tokens": 3}, 400, "context_limit"),
                     ({"prompt": "", "max_tokens": True}, 400, "invalid_max_tokens"),
                     ({"prompt": "", "max_tokens": 5}, 400, "invalid_max_tokens"),
                     ({"prompt": "", "seed": -1}, 400, "invalid_seed"),
                     ({"prompt": "", "temperature": -1}, 400, "invalid_temperature"),
                     ({"prompt": "", "unexpected": 1}, 400, "invalid_fields"),
                     ({"prompt": 3}, 400, "prompt_required")]
            for payload, expected_status, expected_code in cases:
                with self.subTest(payload=payload):
                    status, result, _ = request(server, payload)
                    self.assertEqual((status, result["error"]), (expected_status, expected_code))
            status, result, _ = request(server, {"prompt": "", "max_tokens": 0})
            self.assertEqual(status, 200)
            self.assertEqual(result["token_ids"], [])
            self.assertEqual(service.metrics()["cache_bytes"], 0)

    def test_global_cache_cap_rejects_before_allocation(self):
        with running(self.model, self.tokenizer, max_cache_bytes=8) as (server, service):
            status, result, _ = request(server, {"prompt": "", "max_tokens": 1})
            self.assertEqual((status, result["error"]), (503, "cache_capacity"))
            self.assertEqual(service.metrics()["cache_bytes"], 0)

    def test_fixed_worker_admission_returns_busy_without_spawning_threads(self):
        entered, release = threading.Event(), threading.Event()
        original = self.model.decode
        def slow_decode(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test gate timed out")
            return original(*args, **kwargs)
        with running(self.model, self.tokenizer, workers=1, queue_size=0) as (server, service):
            with mock.patch.object(self.model, "decode", side_effect=slow_decode):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    first = executor.submit(request, server, {"prompt": "", "max_tokens": 1})
                    self.assertTrue(entered.wait(1))
                    try:
                        status, result, _ = request(server, path="/health", method="GET")
                        self.assertEqual((status, result["error"]), (503, "server_busy"))
                        self.assertEqual(len(server._workers), 1)
                        self.assertEqual(service.metrics()["active_connections"], 1)
                    finally:
                        release.set()
                    self.assertEqual(first.result(timeout=2)[0], 200)

    def test_generation_error_releases_cache_and_hides_internal_details(self):
        created = []
        def track_cache(*args, **kwargs):
            cache = PagedLatentCache(*args, **kwargs)
            created.append(cache)
            return cache
        with running(self.model, self.tokenizer) as (server, service):
            with mock.patch("rawllm.serving.PagedLatentCache", side_effect=track_cache):
                with mock.patch.object(self.model, "decode", side_effect=RuntimeError("/private/secret-token-file")):
                    status, payload, _ = request(server, {"prompt": "", "max_tokens": 1})
            self.assertEqual(status, 500)
            self.assertEqual(payload, {"error": "internal_error"})
            self.assertEqual(service.metrics()["cache_bytes"], 0)
            self.assertEqual(created[0].tables, {})
            self.assertEqual(len(created[0].free), created[0].max_pages)
            self.assertEqual(request(server, {"prompt": "", "max_tokens": 1})[0], 200)

    def test_queue_has_finite_capacity_and_waiting_requests_complete(self):
        entered, release = threading.Event(), threading.Event()
        original = self.model.decode
        def slow_decode(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test gate timed out")
            return original(*args, **kwargs)
        with running(self.model, self.tokenizer, workers=1, queue_size=1) as (server, service):
            with mock.patch.object(self.model, "decode", side_effect=slow_decode):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first = executor.submit(request, server, {"prompt": "", "max_tokens": 1})
                    self.assertTrue(entered.wait(1))
                    second = executor.submit(request, server, {"prompt": "", "max_tokens": 1})
                    eventually(lambda: service.metrics()["queued_connections"] == 1)
                    try:
                        self.assertEqual(request(server, path="/health", method="GET")[0], 503)
                        self.assertEqual(service.metrics()["active_connections"], 1)
                        self.assertEqual(service.metrics()["queued_connections"], 1)
                    finally:
                        release.set()
                    self.assertEqual(first.result(timeout=2)[0], 200)
                    self.assertEqual(second.result(timeout=2)[0], 200)

    def test_shutdown_interrupts_idle_client_and_joins_all_workers(self):
        service = InferenceService(self.model, self.tokenizer, TOKEN, ServingLimits(io_timeout=2))
        server = BoundedInferenceServer(("127.0.0.1", 0), service)
        acceptor = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        acceptor.start()
        connection = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2)
        try:
            connection.sendall(b"GET /health HTTP/1.1\r\n")
            eventually(lambda: service.metrics()["active_connections"] == 1)
            started = time.monotonic()
            server.shutdown()
            server.server_close()
            acceptor.join(timeout=1)
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(acceptor.is_alive())
            self.assertTrue(all(not worker.is_alive() for worker in server._workers))
            self.assertEqual(service.metrics()["active_connections"], 0)
            self.assertEqual(service.metrics()["cache_bytes"], 0)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()

    def test_failed_bind_does_not_leak_worker_threads(self):
        with running(self.model, self.tokenizer) as (existing, _):
            service = InferenceService(self.model, self.tokenizer, TOKEN)
            threads_before = {thread.ident for thread in threading.enumerate()}
            with self.assertRaises(OSError):
                BoundedInferenceServer(existing.server_address, service)
            self.assertEqual({thread.ident for thread in threading.enumerate()}, threads_before)

    def test_disconnected_client_cancels_generation_and_releases_budget(self):
        entered, release = threading.Event(), threading.Event()
        original = self.model.decode
        def slow_decode(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original(*args, **kwargs)
        with running(self.model, self.tokenizer) as (server, service):
            with mock.patch.object(self.model, "decode", side_effect=slow_decode):
                connection = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2)
                payload = b'{"prompt":"x","max_tokens":8}'
                headers = (f"POST /generate HTTP/1.1\r\nHost: 127.0.0.1:{server.server_address[1]}\r\n"
                           f"Authorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\n"
                           f"Content-Length: {len(payload)}\r\n\r\n").encode()
                connection.sendall(headers + payload)
                self.assertTrue(entered.wait(1))
                connection.close()
                release.set()
                eventually(lambda: service.metrics()["active_connections"] == 0)
            self.assertEqual(service.metrics()["cache_bytes"], 0)
            self.assertGreaterEqual(service.metrics()["cancelled_requests"], 1)
            self.assertEqual(service.metrics()["completed_generations"], 0)

    def test_io_and_compute_deadlines_release_admission(self):
        with running(self.model, self.tokenizer, io_timeout=0.05, request_timeout=0.08, workers=1, queue_size=0) as (server, service):
            with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=2) as connection:
                connection.sendall(b"GET /health HTTP/1.1\r\n")
                response = connection.recv(4096)
                self.assertIn(b" 408 ", response)
            eventually(lambda: service.metrics()["active_connections"] == 0)
            original = self.model.decode
            def delayed(*args, **kwargs):
                time.sleep(0.1)
                return original(*args, **kwargs)
            with mock.patch.object(self.model, "decode", side_effect=delayed):
                status, result, _ = request(server, {"prompt": "", "max_tokens": 1})
                self.assertEqual(status, 408)
                self.assertEqual(result["error"], "request_deadline")
            self.assertEqual(service.metrics()["cache_bytes"], 0)

    def test_host_rebinding_and_nonloopback_binding_are_rejected(self):
        service = InferenceService(self.model, self.tokenizer, TOKEN)
        for host in ("0.0.0.0", "192.0.2.1", "example.com"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                BoundedInferenceServer((host, 0), service)
        with running(self.model, self.tokenizer) as (server, _):
            status, result = raw_request(server, b"GET /health HTTP/1.1\r\nHost: attacker.example\r\n\r\n")
            self.assertEqual((status, result["error"]), (403, "invalid_host"))


class TokenFileTests(unittest.TestCase):
    def test_create_reuse_environment_and_permission_rejection(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "serve.token"
            token, source = load_bearer_token(path)
            self.assertEqual(source, "generated_file")
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_bearer_token(path), (token, "file"))
            os.environ["RAWLLM_API_TOKEN"] = TOKEN
            self.assertEqual(load_bearer_token(path), (TOKEN, "environment"))
            del os.environ["RAWLLM_API_TOKEN"]
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                load_bearer_token(path)

    def test_symlink_and_short_token_rejection(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "real.token"
            path.write_text(TOKEN)
            path.chmod(0o600)
            linked = Path(directory) / "linked.token"
            linked.symlink_to(path)
            with self.assertRaises(OSError):
                load_bearer_token(linked)
            os.environ["RAWLLM_API_TOKEN"] = "short"
            with self.assertRaises(ValueError):
                load_bearer_token(path)

    def test_fifo_token_file_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "fifo.token"
            os.mkfifo(path, mode=0o600)
            started = time.monotonic()
            with self.assertRaises(ValueError):
                load_bearer_token(path)
            self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":
    unittest.main()
