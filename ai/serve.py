"""Serve a saved local run with bounded workers, bearer auth, and paged inference."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path
import signal
import threading

from rawllm.serving import ServingLimits, load_bearer_token, server_from_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--token-file", type=Path, help="Private bearer-token file; defaults to RUN/serve.token")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--queue", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--cache-mib", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()
    server = None
    try:
        limits = ServingLimits(workers=args.workers, queue_size=args.queue,
                               max_new_tokens=args.max_tokens, max_prompt_tokens=args.max_prompt_tokens,
                               max_cache_bytes=args.cache_mib * 1024**2, request_timeout=args.request_timeout)
        token, source = load_bearer_token(args.token_file or args.run / "serve.token")
        server = server_from_run(args.run, token, (args.host, args.port), limits)
    except (OSError, ValueError, TypeError, KeyError, MemoryError):
        parser.exit(2, "Unable to start: verify the trusted run, private token file, loopback address, and resource limits.\n")
    shutdown_started = threading.Event()
    def stop(signum, frame):
        if not shutdown_started.is_set():
            shutdown_started.set()
            server.service.stop_event.set()
            threading.Thread(target=server.shutdown, name="rawllm-shutdown", daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({"status": "listening", "host": server.server_address[0], "port": server.server_address[1],
                      "authentication": "bearer_required", "token_source": source,
                      "workers": limits.workers, "queue": limits.queue_size}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
