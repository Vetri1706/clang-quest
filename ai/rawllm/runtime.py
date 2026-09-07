"""POSIX local run ownership, cooperative stopping and process supervision."""
from contextlib import AbstractContextManager
import fcntl
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".json-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunLease(AbstractContextManager):
    """Nonblocking advisory lock, automatically released by the kernel on death.

    Never unlink the lock file: replacing its inode would split ownership. This
    coordinates cooperating processes on a local POSIX filesystem, not NFS or
    hostile writers. Metadata is diagnostic; kernel lock ownership is decisive.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.descriptor = None
        self.run_id = uuid.uuid4().hex

    def __enter__(self):
        if self.descriptor is not None:
            raise RuntimeError("Run lease is already held")
        self.directory.mkdir(parents=True, exist_ok=True)
        parent = os.open(self.directory.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        descriptor = os.open(self.directory / ".run.lock",
                             os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Run lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process owns this run directory") from error
            metadata = json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                   "run_id": self.run_id, "acquired_unix": time.time()}).encode()
            os.ftruncate(descriptor, 0)
            os.write(descriptor, metadata)
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, *exc):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


class StopRequest(AbstractContextManager):
    """Signal handlers only set a flag; callers commit at an iteration boundary."""
    def __init__(self):
        self.requested = False
        self.signal_number = None
        self._previous = {}

    def _handle(self, number, frame):
        self.requested = True
        self.signal_number = number

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Signal management requires the main thread")
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *exc):
        for number, previous in self._previous.items():
            signal.signal(number, previous)


def reconcile_log(path, committed_step, max_line_bytes=65536):
    """Drop uncommitted/truncated tail rows; checkpoint counters are authoritative.

    A committed checkpoint can exist without its final telemetry row if the
    process died between checkpoint publication and log append. Missing rows are
    not invented. Malformed data before later committed rows fails explicitly.
    """
    path = Path(path)
    if not path.exists():
        return
    temporary = None
    try:
        with path.open("rb") as source, tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            last = 0
            tail = False
            while True:
                row = source.readline(max_line_bytes + 1)
                if not row:
                    break
                if len(row) > max_line_bytes:
                    raise ValueError("Training log row exceeds its byte limit")
                try:
                    record = json.loads(row)
                    step = record["step"]
                except (ValueError, TypeError, KeyError):
                    tail = True
                    continue
                if type(step) is not int or step < 1:
                    raise ValueError("Invalid training log step")
                if step > committed_step:
                    tail = True
                    continue
                if tail or step <= last:
                    raise ValueError("Training log contains corruption before its committed tail")
                if not row.endswith(b"\n"):
                    tail = True
                    continue
                out.write(row)
                last = step
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def supervise(command, *, cwd, log_path, max_seconds=90, max_output_bytes=2 * 1024**2, grace_seconds=5):
    """Bound wall time/output and clean up a new POSIX process group.

    This is a resource watchdog for trusted framework commands, not a code
    sandbox. It supplies one BLAS thread and lower scheduling priority. It does
    not claim an OS resident-memory limit or intercept native kernel execution.
    """
    if not command or any(not isinstance(s, str) or not s for s in command):
        raise ValueError("Command must be a nonempty argument vector")
    if not all(math.isfinite(x) for x in (max_seconds, grace_seconds)) or not 0 < max_seconds <= 600 or not 0 <= grace_seconds <= 30:
        raise ValueError("Use 0..600 seconds and 0..30 seconds shutdown grace")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 16 * 1024**2:
        raise ValueError("Output limit must be between 1 byte and 16 MiB")
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + max_seconds
    written = 0
    reason = "completed"
    termination = None
    process = None

    def notify_group(number):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            return

    with StopRequest() as stop, log_path.open("xb") as log, selectors.DefaultSelector() as selector:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                os.setpriority(os.PRIO_PROCESS, process.pid, 10)
            except (ProcessLookupError, PermissionError):
                pass
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if termination is None:
                    if stop.requested:
                        reason = "cancelled"
                    elif now >= deadline:
                        reason = "deadline"
                    elif process.poll() is not None and selector.get_map():
                        # A descendant can inherit stdout after the leader exits.
                        reason = "completed" if process.returncode == 0 else "failed"
                        termination = now
                        notify_group(signal.SIGTERM)
                    if reason in ("cancelled", "deadline", "output_limit"):
                        termination = now
                        notify_group(signal.SIGTERM)
                if termination is not None and now - termination >= grace_seconds:
                    notify_group(signal.SIGKILL)
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = max_output_bytes - written
                    log.write(chunk[:remaining])
                    written += min(remaining, len(chunk))
                    if len(chunk) > remaining and reason not in ("cancelled", "deadline"):
                        reason = "output_limit"
                        if termination is None:
                            termination = time.monotonic()
                            notify_group(signal.SIGTERM)
                if termination is not None and time.monotonic() - termination > grace_seconds + 2:
                    break
            process.wait(timeout=2)
            if reason == "completed" and process.returncode:
                reason = "failed"
        finally:
            if process is not None:
                notify_group(signal.SIGKILL)
                process.wait(timeout=2)
                process.stdout.close()
            log.flush()
            os.fsync(log.fileno())
    return {"reason": reason, "returncode": process.returncode, "seconds": time.monotonic() - started,
            "output_bytes": written, "max_output_bytes": max_output_bytes, "max_seconds": max_seconds,
            "blas_threads": 1, "memory_limit_enforced": False}
