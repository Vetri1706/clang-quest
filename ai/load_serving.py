"""Run a tiny bounded functional load probe against an isolated local server.

The target is always a newly started 127.0.0.1 server, with two workers and two
queue slots. This validates deterministic responses and overload handling. It
is not a sustained throughput benchmark, capacity certification, or model
quality evaluation. Credentials live only in a temporary private directory.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from generate import generate, load_run
from rawllm.runtime import atomic_json
from rawllm.safeio import read_json
from rawllm.serving import load_bearer_token


ROOT = Path(__file__).resolve().parent
SERVER_LIFETIME_SECONDS = 20.0


class _ReferenceTokenizer:
    """Capture the tokens from generate.py's one reference decoding pass."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id, self.bos_id, self.eos_id = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id
        self.tokens = None

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, tokens):
        self.tokens = list(tokens)
        return self.tokenizer.decode(self.tokens)


def _percentiles(latencies):
    if not latencies:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None}
    values = np.percentile(np.asarray(latencies, dtype=np.float64) * 1000, [50, 95, 99])
    return dict(zip(("p50_ms", "p95_ms", "p99_ms"), map(float, values)))


def run_probe(run, requests=64, clients=4, tokens=8):
    if type(requests) is not int or not 1 <= requests <= 256:
        raise ValueError("The local probe accepts only 1..256 requests")
    if type(clients) is not int or not 1 <= clients <= 8:
        raise ValueError("The local probe accepts only 1..8 client threads")
    if type(tokens) is not int or not 1 <= tokens <= 16:
        raise ValueError("The local probe accepts only 1..16 generated tokens per request")
    run = Path(run).resolve()
    manifest_before = read_json(run / "checkpoint/manifest.json")
    settings_before = read_json(run / "settings.json")
    model, tokenizer = load_run(run)
    if model.config.parameter_count > 1_000_000:
        raise ValueError("This functional probe is intentionally limited to models below one million parameters")
    prompt, seed, temperature = "C++", 23, 0.8
    reference_tokenizer = _ReferenceTokenizer(tokenizer)
    expected = generate(model, reference_tokenizer, prompt, tokens, temperature, seed)
    expected_ids = reference_tokenizer.tokens
    parameter_count = model.config.parameter_count
    parameter_bytes = sum(parameter.data.nbytes for parameter in model.params.values())
    del model
    if read_json(run / "checkpoint/manifest.json") != manifest_before or read_json(run / "settings.json") != settings_before:
        raise ValueError("Run artifacts changed during reference generation")
    records, metrics = [], None
    process = None
    watchdog = None
    watchdog_fired = threading.Event()
    probe_error = None
    stderr_bytes = 0
    shutdown_seconds = None
    server_started = None
    load_started = None
    elapsed = None
    with tempfile.TemporaryDirectory(prefix="rawllm-load-") as temporary:
        token_path = Path(temporary) / "bearer.token"
        # Use a deliberately empty environment key so a user's interactive
        # RAWLLM_API_TOKEN is never copied into this disposable probe.
        private_environment_key = "RAWLLM_LOAD_UNUSED_" + os.urandom(8).hex()
        token, _ = load_bearer_token(token_path, environment=private_environment_key)
        environment = os.environ.copy()
        environment.pop("RAWLLM_API_TOKEN", None)
        try:
            server_started = time.monotonic()
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "serve.py"), str(run), "--host", "127.0.0.1", "--port", "0",
                 "--workers", "2", "--queue", "2", "--request-timeout", "10", "--token-file", str(token_path)],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

            def hard_stop():
                if process.poll() is None:
                    watchdog_fired.set()
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            watchdog = threading.Timer(SERVER_LIFETIME_SECONDS, hard_stop)
            watchdog.daemon = True
            watchdog.start()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=5):
                    raise RuntimeError("startup_timeout")
                startup_line = process.stdout.readline(4097)
            if len(startup_line) > 4096 or not startup_line.endswith(b"\n"):
                raise RuntimeError("invalid_startup_response")
            startup = json.loads(startup_line)
            if startup.get("host") != "127.0.0.1" or startup.get("workers") != 2 or startup.get("queue") != 2:
                raise RuntimeError("invalid_server_configuration")
            port = startup["port"]
            if type(port) is not int or not 1 <= port <= 65535:
                raise RuntimeError("invalid_server_port")
            deadline = server_started + SERVER_LIFETIME_SECONDS
            body = json.dumps({"prompt": prompt, "max_tokens": tokens, "temperature": temperature, "seed": seed}).encode("utf-8")
            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}

            def issue(index):
                started = time.monotonic()
                if started >= deadline:
                    return {"index": index, "status": "deadline", "latency_seconds": 0.0}
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=max(0.05, min(5, deadline - started)))
                try:
                    connection.request("POST", "/generate", body=body, headers=headers)
                    response = connection.getresponse()
                    raw = response.read(65537)
                    if len(raw) > 65536:
                        return {"index": index, "status": "oversized_response", "latency_seconds": time.monotonic() - started}
                    payload = json.loads(raw)
                    if response.status == 200:
                        valid = (payload.get("generated_text") == expected and payload.get("token_ids") == expected_ids
                                 and payload.get("generated_tokens") == len(expected_ids) and payload.get("seed") == seed)
                        status = "success" if valid else "generation_mismatch"
                    elif response.status == 503 and payload.get("error") == "server_busy":
                        status = "saturated"
                    else:
                        status = "unexpected_http_status"
                    return {"index": index, "status": status, "http_status": response.status,
                            "latency_seconds": time.monotonic() - started}
                except (OSError, http.client.HTTPException, ValueError):
                    return {"index": index, "status": "transport_or_json_error", "latency_seconds": time.monotonic() - started}
                finally:
                    connection.close()

            load_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=clients, thread_name_prefix="rawllm-load-client") as executor:
                records = list(executor.map(issue, range(requests)))
            elapsed = time.monotonic() - load_started
            # A successful body may reach the client a few microseconds before
            # the server releases its connection slot. Poll boundedly until all
            # generation workers have unwound, counting /metrics itself as one.
            metrics_deadline = min(deadline, time.monotonic() + 2)
            while time.monotonic() < metrics_deadline:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    connection.request("GET", "/metrics")
                    response = connection.getresponse()
                    payload = json.loads(response.read(65536))
                    if response.status == 200:
                        metrics = payload
                        if payload["cache_bytes"] == 0 and payload["queued_connections"] == 0 and payload["active_connections"] <= 1:
                            break
                finally:
                    connection.close()
                time.sleep(0.01)
            if metrics is None:
                raise RuntimeError("metrics_unavailable")
            if read_json(run / "checkpoint/manifest.json") != manifest_before or read_json(run / "settings.json") != settings_before:
                raise RuntimeError("run_changed_during_probe")
        except (OSError, ValueError, KeyError, RuntimeError, http.client.HTTPException):
            probe_error = "probe_failed"
        finally:
            shutdown_started = time.monotonic()
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    stdout_tail, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    stdout_tail, stderr = process.communicate(timeout=2)
                stderr_bytes = len(stderr)
                if stdout_tail:
                    probe_error = "unexpected_server_output"
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join(timeout=1)
            shutdown_seconds = time.monotonic() - shutdown_started
    successes = [record for record in records if record["status"] == "success"]
    saturated = [record for record in records if record["status"] == "saturated"]
    failures = [record for record in records if record["status"] not in ("success", "saturated")]
    cache_clean = metrics is not None and metrics.get("cache_bytes") == 0
    server_stopped = process is not None and process.poll() is not None
    passed = (probe_error is None and len(records) == requests and len(successes) > 0 and not failures
              and cache_clean and server_stopped and process.returncode == 0 and stderr_bytes == 0
              and not watchdog_fired.is_set())
    return {"format": "rawllm-local-serving-load-probe", "version": 1, "status": "pass" if passed else "fail",
            "scope": "tiny_bounded_functional_load_probe_not_a_sustained_benchmark",
            "run_name": run.name, "parameter_count": parameter_count, "model_parameter_array_bytes": parameter_bytes,
            "checkpoint_payload_sha256": manifest_before["sha256"], "tokenizer_sha256": settings_before["tokenizer_sha256"],
            "requested_requests": requests, "completed_requests": len(records), "successful_requests": len(successes),
            "saturated_503_requests": len(saturated), "failed_requests": len(failures),
            "failure_categories": {kind: sum(row["status"] == kind for row in failures) for kind in sorted({row["status"] for row in failures})},
            "probe_error": probe_error, "client_threads": clients, "server_workers": 2, "server_queue_capacity": 2,
            "requested_tokens_per_request": tokens, "actual_reference_generated_tokens": len(expected_ids),
            "deterministic_reference_evaluations": 1, "all_successful_generations_match_reference": bool(successes) and not any(row["status"] == "generation_mismatch" for row in records),
            "reference_token_ids_sha256": hashlib.sha256(json.dumps(expected_ids).encode("ascii")).hexdigest(),
            "sampling_seed": seed, "temperature": temperature, "prompt": prompt,
            "load_elapsed_seconds": elapsed, "successful_request_latency": _percentiles([row["latency_seconds"] for row in successes]),
            "all_request_latency": _percentiles([row["latency_seconds"] for row in records]),
            "server_metrics_after_load": metrics, "cache_bytes_after_load": None if metrics is None else metrics.get("cache_bytes"),
            "cache_budget_excludes_model_and_allocator_overhead": True, "server_hard_timeout_seconds": SERVER_LIFETIME_SECONDS,
            "watchdog_fired": watchdog_fired.is_set(), "shutdown_seconds": shutdown_seconds,
            "server_exit_code": None if process is None else process.returncode, "server_stderr_bytes": stderr_bytes,
            "server_stopped": server_stopped, "token_file_removed": not token_path.exists(), "blas_threads": 1,
            "numpy_version": np.__version__, "model_quality_assessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/hardened_dpo")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/serving_load.json")
    args = parser.parse_args()
    try:
        report = run_probe(args.run, args.requests, args.clients, args.tokens)
        atomic_json(args.output, report)
    except (OSError, ValueError, TypeError, KeyError, MemoryError):
        parser.exit(2, "Load probe failed: verify the trusted tiny run, output path, and bounded arguments.\n")
    print(json.dumps({"status": report["status"], "successful_requests": report["successful_requests"],
                      "saturated_503_requests": report["saturated_503_requests"], "failed_requests": report["failed_requests"],
                      "latency": report["successful_request_latency"], "server_stopped": report["server_stopped"]}))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
