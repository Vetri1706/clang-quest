"""Bounded, fail-closed C++20 runner for a local macOS teaching dashboard.

Only standard-library Python and Apple's local command-line tools are used.
Seatbelt is an OS execution boundary, not a claim of safe Internet multitenancy.
The parent never uses preexec_fn: a separate trusted helper installs rlimits.
"""
from __future__ import annotations

import ctypes
import functools
import json
import math
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import sys
import tempfile
import time

SOURCE_BYTES = 32 * 1024
INPUT_BYTES = 16 * 1024
OUTPUT_BYTES = 16 * 1024
MAX_CASES = 12
COMPILE_SECONDS = 15.0
RUN_SECONDS = 2.0
STARTUP_SECONDS = 10.0
COMPILE_MEMORY_BYTES = 512 * 1024 * 1024
RUN_MEMORY_BYTES = 256 * 1024 * 1024
SANDBOX = Path('/usr/bin/sandbox-exec')


class RunnerUnavailable(RuntimeError):
    """The required local execution boundary could not be established."""


class _TaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        'virtual_size', 'resident_size', 'total_user', 'total_system',
        'threads_user', 'threads_system')] + [
        (name, ctypes.c_int32) for name in (
            'policy', 'faults', 'pageins', 'cow_faults', 'messages_sent',
            'messages_received', 'syscalls_mach', 'syscalls_unix', 'csw',
            'threadnum', 'numrunning', 'priority')]


class _ProcessMonitor:
    """Read process-group RSS without shell commands or a Python dependency."""
    def __init__(self):
        try:
            self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            self.lib.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                               ctypes.c_void_p, ctypes.c_int]
            self.lib.proc_listpids.restype = ctypes.c_int
            self.lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int,
                                              ctypes.c_uint64, ctypes.c_void_p,
                                              ctypes.c_int]
            self.lib.proc_pidinfo.restype = ctypes.c_int
            self.lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            self.lib.proc_pidpath.restype = ctypes.c_int
            self._unreadable = {}
        except (OSError, AttributeError) as exc:
            raise RunnerUnavailable('macOS process monitoring is unavailable') from exc

    def executable(self, pid):
        path = ctypes.create_string_buffer(4096)
        size = self.lib.proc_pidpath(pid, path, len(path))
        return os.fsdecode(path.value) if size > 0 else None

    def sample(self, pgid):
        pids = (ctypes.c_int * 128)()
        size = self.lib.proc_listpids(2, pgid, pids, ctypes.sizeof(pids))
        if size < 0 or size >= ctypes.sizeof(pids):
            raise RunnerUnavailable('process-group accounting failed')
        count = size // ctypes.sizeof(ctypes.c_int)
        rss, threads = 0, 0
        for pid in pids[:count]:
            if pid <= 0:
                continue
            info = _TaskInfo()
            read = self.lib.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
            if read == ctypes.sizeof(info):
                rss += info.resident_size
                threads += info.threadnum
                self._unreadable.pop(pid, None)
            elif read == 0:
                self._unreadable[pid] = self._unreadable.get(pid, 0) + 1
                if self._unreadable[pid] >= 3:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        continue
                    raise RunnerUnavailable('A running process could not be memory-accounted')
            elif read != 0:
                raise RunnerUnavailable('process memory accounting failed')
        return rss, count, threads


@functools.lru_cache(maxsize=1)
def _toolchain():
    if sys.platform != 'darwin' or not SANDBOX.is_file():
        raise RunnerUnavailable('C++ execution requires macOS sandbox-exec; execution is disabled')
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'}
    try:
        compiler = Path(subprocess.check_output(
            ['/usr/bin/xcrun', '--find', 'clang++'], env=env,
            stderr=subprocess.DEVNULL, timeout=5, text=True).strip())
        sdk = Path(subprocess.check_output(
            ['/usr/bin/xcrun', '--show-sdk-path'], env=env,
            stderr=subprocess.DEVNULL, timeout=5, text=True).strip()).resolve()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunnerUnavailable('Install the Apple C++ command-line tools to enable execution') from exc
    if not compiler.is_file() or not sdk.is_dir():
        raise RunnerUnavailable('The selected Apple C++ toolchain is incomplete')
    # Retain the clang++ symlink in argv[0] so the driver links the C++ standard library.
    tool_root = compiler.parent.parent.parent.resolve()
    if not (str(tool_root).startswith('/Library/Developer/') or
            str(tool_root).startswith('/Applications/')):
        raise RunnerUnavailable('The selected compiler is outside an Apple toolchain directory')
    _ProcessMonitor()
    return compiler, sdk, tool_root


def _quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def _profile(job: Path, compiler: Path, sdk: Path, tool_root: Path, compile_phase: bool):
    binary = job / 'program'
    reads = [Path('/System/Library'), Path('/usr/lib'), Path('/private/var/db/dyld'), job]
    if compile_phase:
        reads.extend([tool_root, sdk])
    metadata = {Path('/')}
    for path in reads + [compiler if compile_phase else binary]:
        metadata.update(path.parents)
    # Literal root access is needed by Apple's loader/clang. It does not recurse.
    rules = ['(version 1)', '(deny default)', '(deny process-info*)',
             '(allow process-info* (target self))',
             '(allow sysctl-read (sysctl-name-prefix "hw.") '
             '(sysctl-name-prefix "kern.os") (sysctl-name "kern.argmax") '
             '(sysctl-name "kern.secure_kernel") (sysctl-name "kern.hv_vmm_present") '
             '(sysctl-name "sysctl.proc_native"))',
             '(allow file-read* (literal "/"))',
             '(allow file-read-metadata ' + ' '.join(
                 '(literal ' + _quote(p) + ')' for p in sorted(metadata)) + ')',
             '(allow file-read* ' + ' '.join(
                 '(subpath ' + _quote(p) + ')' for p in reads) +
             ' (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))']
    if compile_phase:
        rules.extend([
            '(allow file-write* (subpath ' + _quote(job) + ') (literal "/dev/null"))',
            '(allow process-exec (subpath ' + _quote(tool_root / 'usr/bin') + '))',
            '(allow process-fork)'])
    else:
        rules.append('(allow process-exec (literal ' + _quote(binary) + '))')
    # No network, Mach service lookup, process-fork, or file-write grants at runtime.
    return '\n'.join(rules)


def _child(config):
    """Trusted process entry point: set limits before any untrusted executable."""
    try:
        cpu = int(math.ceil(config['seconds']))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024 if config['compile'] else 0,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        # macOS rejects RLIMIT_AS/DATA/RSS changes on this host. Memory is
        # enforced by the parent's process-group RSS monitor every 20 ms.
        # RLIMIT_NPROC is per user on macOS. Seatbelt's runtime fork denial and
        # parent-side group monitoring supply the stricter per-job process limit.
        resource.setrlimit(resource.RLIMIT_NPROC, (1024, 1024))
        if config.get('ready_fd') is not None:
            os.dup2(config['ready_fd'], 3, inheritable=True)
            if config['ready_fd'] != 3:
                os.close(config['ready_fd'])
        os.nice(10)
        os.execv(str(SANDBOX), [str(SANDBOX), '-p', config['profile'], *config['command']])
    except Exception as exc:
        os.write(2, ('Runner setup failed: ' + type(exc).__name__ + '\n').encode())
        os._exit(125)


def _terminate_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        raise RunnerUnavailable('A job did not terminate after cancellation')


def _execute(command, profile, job, stdin, seconds, compile_phase):
    started = time.monotonic()
    memory_limit = COMPILE_MEMORY_BYTES if compile_phase else RUN_MEMORY_BYTES
    config = dict(command=[str(x) for x in command], profile=profile,
                  seconds=seconds, compile=compile_phase, memory_bytes=memory_limit)
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(job), 'TMPDIR': str(job),
           'LANG': 'C', 'LC_ALL': 'C', 'TERM': 'dumb',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'VECLIB_MAXIMUM_THREADS': '1'}
    monitor = _ProcessMonitor()
    ready_read, ready_write = os.pipe() if not compile_phase else (None, None)
    config['ready_fd'] = ready_write
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_child',
                             json.dumps(config)], cwd=job, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True,
                            close_fds=True, bufsize=0,
                            pass_fds=(() if ready_write is None else (ready_write,)))
    if ready_write is not None:
        os.close(ready_write)
    selector = selectors.DefaultSelector()
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    if ready_read is not None:
        os.set_blocking(ready_read, False)
        selector.register(ready_read, selectors.EVENT_READ, 'ready')
    for name in buffers:
        stream = getattr(proc, name)
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    pending = memoryview(stdin)
    os.set_blocking(proc.stdin.fileno(), False)
    if pending:
        selector.register(proc.stdin, selectors.EVENT_WRITE, 'stdin')
    else:
        proc.stdin.close()
    status, peak_rss = 'ok', 0
    program_started = started if compile_phase else None
    deadline = started + (seconds if compile_phase else STARTUP_SECONDS)
    try:
        while selector.get_map():
            now = time.monotonic()
            if now > deadline:
                status = 'time_limit' if program_started is not None else 'startup_limit'
                break
            rss, process_count, threads = monitor.sample(proc.pid)
            peak_rss = max(peak_rss, rss)
            if rss > memory_limit:
                status = 'memory_limit'
                break
            if process_count > (8 if compile_phase else 1) or threads > (64 if compile_phase else 8):
                status = 'process_limit'
                break
            for key, _ in selector.select(timeout=0.02):
                stream, name = key.fileobj, key.data
                if name == 'ready':
                    marker = os.read(stream, 16)
                    selector.unregister(stream)
                    os.close(stream)
                    ready_read = None
                    if marker.startswith(b'R'):
                        program_started = time.monotonic()
                        deadline = program_started + seconds
                    continue
                if name == 'stdin':
                    try:
                        count = os.write(stream.fileno(), pending[:4096])
                        pending = pending[count:]
                    except BrokenPipeError:
                        pending = pending[:0]
                    if not pending:
                        selector.unregister(stream)
                        stream.close()
                else:
                    try:
                        chunk = os.read(stream.fileno(), 4096)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    remaining = OUTPUT_BYTES - len(buffers[name])
                    buffers[name].extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        status = 'output_limit'
                        break
            if status != 'ok':
                break
        if status == 'ok':
            remaining_time = max(0.001, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining_time)
            except subprocess.TimeoutExpired:
                status = 'time_limit'
    finally:
        _terminate_group(proc)
        selector.close()
        if ready_read is not None:
            os.close(ready_read)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not stream.closed:
                stream.close()
    if status == 'ok' and proc.returncode:
        status = 'time_limit' if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL) else 'runtime_error'
    decoded = {name: bytes(value).decode('utf-8', errors='replace').encode('utf-8')[:OUTPUT_BYTES].decode('utf-8', errors='ignore')
               for name, value in buffers.items()}
    for name in decoded:
        decoded[name] = decoded[name].replace(str(job), '<workspace>')
    finished = time.monotonic()
    return dict(status=status, exit_code=proc.returncode,
                time_ms=round((finished - (program_started or started)) * 1000, 2),
                startup_time_ms=round(((program_started or started) - started) * 1000, 2),
                total_time_ms=round((finished - started) * 1000, 2),
                peak_rss_bytes=peak_rss, **decoded)


def _normalized(text):
    return '\n'.join(line.rstrip() for line in text.replace('\r\n', '\n').split('\n')).strip('\n')


def run_cpp(source: str, cases: list[dict]):
    """Compile once and run every case in an independent sandbox process.

    Every case requires string ``input`` and ``output`` values. Results include
    test data; callers MUST remove hidden input/expected/actual before sending
    hidden cases to a browser. The caller should serialize jobs with a semaphore.
    """
    started = time.monotonic()
    if not isinstance(source, str) or len(source.encode('utf-8')) > SOURCE_BYTES:
        raise ValueError('source must be a string of at most 32 KiB')
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError('provide between 1 and 12 cases')
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('each case must be an object')
        for key in ('input', 'output'):
            if not isinstance(case.get(key), str) or len(case[key].encode('utf-8')) > INPUT_BYTES:
                raise ValueError('case input and output must be strings of at most 16 KiB')
    try:
        compiler, sdk, tool_root = _toolchain()
        with tempfile.TemporaryDirectory(prefix='cppquest-') as directory:
            job = Path(directory).resolve()
            (job / 'main.cpp').write_text(source, encoding='utf-8')
            # This small constructor marks entry into user-space initializers,
            # excluding Apple's sometimes slow first-execution validation.
            # The startup allowance is bounded independently; CPU/RSS limits
            # already apply throughout startup, including earlier constructors.
            (job / 'runner_start.cpp').write_text(
                '#include <unistd.h>\n'
                '__attribute__((constructor(101))) static void cppquest_ready(){'
                'const char marker=\'R\'; write(3,&marker,1);close(3);}', encoding='utf-8')
            command = [compiler, '-std=c++20', '-O0', '-Wall', '-Wextra',
                       '-fno-color-diagnostics', '-ferror-limit=8',
                       '-isysroot', sdk, '-fno-modules', '-fno-implicit-modules',
                       '-o', job / 'program', job / 'main.cpp', job / 'runner_start.cpp']
            compilation = _execute(command, _profile(job, compiler, sdk, tool_root, True),
                                   job, b'', COMPILE_SECONDS, True)
            diagnostics = compilation['stderr'] + compilation['stdout']
            if compilation['status'] != 'ok':
                status = 'compile_error' if compilation['status'] == 'runtime_error' else compilation['status']
                if compilation['exit_code'] == 125 or diagnostics.startswith('sandbox-exec:'):
                    status = 'unavailable'
                return dict(compile_status=status, diagnostics=diagnostics,
                            compile_time_ms=compilation['time_ms'], results=[],
                            duration_ms=round((time.monotonic() - started) * 1000, 2))
            runtime_profile = _profile(job, compiler, sdk, tool_root, False)
            results = []
            for index, case in enumerate(cases):
                outcome = _execute([job / 'program'], runtime_profile, job,
                                   case['input'].encode('utf-8'), RUN_SECONDS, False)
                passed = outcome['status'] == 'ok' and _normalized(outcome['stdout']) == _normalized(case['output'])
                status = ('passed' if passed else 'wrong_answer') if outcome['status'] == 'ok' else outcome['status']
                results.append(dict(index=index, status=status, passed=passed,
                                    input=case['input'], expected=case['output'],
                                    actual=outcome['stdout'], stdout=outcome['stdout'],
                                    stderr=outcome['stderr'], time_ms=outcome['time_ms'],
                                    startup_time_ms=outcome['startup_time_ms'],
                                    total_time_ms=outcome['total_time_ms'],
                                    peak_rss_bytes=outcome['peak_rss_bytes'],
                                    exit_code=outcome['exit_code']))
            return dict(compile_status='ok', diagnostics=diagnostics,
                        compile_time_ms=compilation['time_ms'], results=results,
                        duration_ms=round((time.monotonic() - started) * 1000, 2))
    except (RunnerUnavailable, OSError) as exc:
        return dict(compile_status='unavailable', diagnostics=str(exc), results=[],
                    compile_time_ms=0, duration_ms=round((time.monotonic() - started) * 1000, 2))


def compiler_health():
    """Real startup smoke test. Cache the result in the web server, not per request."""
    outcome = run_cpp('#include <iostream>\nint main(){int a,b; std::cin>>a>>b;std::cout<<a+b;}',
                      [{'input': '19 23\n', 'output': '42'}])
    return dict(available=outcome['compile_status'] == 'ok' and
                bool(outcome['results']) and outcome['results'][0]['passed'],
                engine='Apple clang++ / C++20 / macOS Seatbelt',
                sandbox='deny by default; no runtime network, file writes, or process fork',
                limits=dict(source_bytes=SOURCE_BYTES, stdin_bytes=INPUT_BYTES,
                            stdout_bytes=OUTPUT_BYTES, stderr_bytes=OUTPUT_BYTES,
                            compile_seconds=COMPILE_SECONDS, run_seconds=RUN_SECONDS,
                            startup_seconds=STARTUP_SECONDS,
                            compile_rss_bytes=COMPILE_MEMORY_BYTES, run_rss_bytes=RUN_MEMORY_BYTES,
                            max_cases=MAX_CASES),
                details=outcome)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '_child':
        _child(json.loads(sys.argv[2]))
    else:
        print(json.dumps(compiler_health(), indent=2))
