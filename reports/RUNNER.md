# Local C++ execution engine

`runner.py` compiles real C++20 with the selected Apple clang++ toolchain and runs each case in a separate macOS Seatbelt process. It uses Python's standard library only. It never runs submitted code without the sandbox, never invokes a shell, and never uses Docker. The compiler and runtime have separate deny-by-default profiles.

```python
from runner import compiler_health, run_cpp

health = compiler_health()  # Run once when the service starts; cache in the service.
result = run_cpp(
    '#include <iostream>\nint main(){int a,b;std::cin>>a>>b;std::cout<<a+b;}',
    [{'input': '19 23\n', 'output': '42\n'}],
)
```

The web service must serialize jobs with one semaphore. `run_cpp` compiles once and runs up to 12 cases. It returns `compile_status`, `diagnostics`, `compile_time_ms`, `results`, and `duration_ms`. Each result has `status`, `passed`, `input`, `expected`, `actual`, `stdout`, `stderr`, `time_ms`, `startup_time_ms`, `total_time_ms`, `peak_rss_bytes`, and `exit_code`. Success statuses are `passed` and `wrong_answer`; failures include `time_limit`, `startup_limit`, `memory_limit`, `process_limit`, `output_limit`, and `runtime_error`. Compilation uses `ok`, `compile_error`, resource statuses, or `unavailable`. Invalid request schemas raise `ValueError` before execution. Output comparison normalizes CRLF, trailing horizontal whitespace, and outer blank lines; internal whitespace remains significant.

Results contain all test data for grading. The web service must remove hidden-case `input`, `expected`, `actual`, `stdout`, and `stderr` before returning them to the browser. Submitted programs can echo their inputs, so hiding only the `input` field is insufficient. Hidden results should expose only the case index, pass/fail status, and permitted aggregate timing. Protect the local API against cross-origin requests and require its local session token; this module does not implement HTTP authentication.

## Bounds

| Resource | Compilation | Each test process |
|---|---:|---:|
| Source | 32 KiB, UTF-8 | Same binary, compiled once |
| Stdin / expected output | No compiler input | 16 KiB each |
| Captured stdout / stderr | 16 KiB each | 16 KiB each |
| Wall clock | 15 seconds including startup | 2 seconds after readiness, plus at most 10 seconds startup |
| CPU soft / hard | 15 / 16 seconds | 2 / 3 seconds, including startup |
| Sampled process-group RSS | 512 MiB | 256 MiB |
| Process-group count | 8 | 1; Seatbelt also denies fork/spawn |
| Sampled thread count | 64 | 8 |
| Open descriptors | 64 | 64 |
| File size | 8 MiB | 0 |
| Core dumps | Disabled | Disabled |

The parent polls every 20 ms and kills the entire child process group on a violated bound or completion. RSS and thread/process counts are sampled limits: a process can temporarily overshoot between samples. macOS on this host rejects changes to `RLIMIT_AS`, `RLIMIT_DATA`, and `RLIMIT_RSS`; the runner therefore uses the public `libproc` accounting APIs and does **not** claim a kernel-enforced address-space ceiling. File/CPU/descriptor limits use `setrlimit`. `RLIMIT_NPROC` is also set to 1024, but it is a per-user ceiling; runtime Seatbelt fork denial and per-group monitoring provide the job-specific process restriction.

Apple's first execution validation sometimes takes several seconds on this machine. A small separate C++ translation unit supplies a constructor that writes one readiness byte to a dedicated inherited pipe and immediately closes it. The 2-second execution timer starts at that marker. The 10-second startup deadline, CPU limit, RSS monitor, output bounds, and sandbox already apply before the marker, including to a deliberately earlier user constructor. Thus a submitted constructor cannot create an unbounded pre-main phase. Runtime duration excludes measured startup; total duration includes it. No compiler optimization is requested (`-O0`). Child niceness is increased by 10 to reduce disruption.

## Access boundary

The compiler may read the selected Apple toolchain, SDK, system libraries, its own job directory, a small set of non-private system facts, and null/random devices. It may write only its job directory and `/dev/null`. It may execute only tools in the selected toolchain's `usr/bin` directory. Implicit Clang modules are disabled. Submitted `#include` paths cannot grant additional filesystem rights.

The runtime can read system libraries, its job directory, null/random devices, and narrowly filtered hardware/OS sysctls. It has no network, Mach service lookup, filesystem-write, or fork grant. Process-information access is explicitly denied except for the current process; macOS requires this extra rule even with a default-deny profile. It can execute only its own compiled binary. System access is restricted to `/System/Library`, not the broader `/System` hierarchy, whose Data volume can alias private user files. Literal root-directory reads and metadata for explicit ancestors are needed by the Apple loader; these permissions do not recursively expose home directories. The inherited environment contains only explicit PATH, HOME/TMPDIR pointing at the job directory, locale settings, and single-thread library hints. No parent credentials or environment secrets are inherited. The helper starts in a new session, closes unrelated descriptors, installs limits, and executes sandbox-exec. No `preexec_fn` is used, so the service can call it from a Python worker thread.

`sandbox-exec` is deprecated in Apple's installed manual, and Seatbelt profile syntax is a private platform interface. This implementation is suitable for a local learning application with defense in depth and tested failure behavior. It is not a supported Internet multi-tenant sandbox. A host OS update may change its behavior; run the smoke test and negative suite after updates, and disable execution if they fail. Toolchain discovery, profile installation, process monitoring, or limit setup failures do not fall back to unsandboxed execution.

## Verification

```sh
/usr/bin/python3 -m unittest discover -s work/dashboard-runtime -p 'test_runner.py' -v
/usr/bin/python3 work/dashboard-runtime/runner.py
```

The suite compiles and executes real C++ programs. It checks arithmetic and wrong answers, private includes, home-file reads and Data-volume aliases, network connection attempts, file writes, fork and spawn, parent process arguments, signal permission, environment isolation, infinite loops, stdout/stderr floods, memory exhaustion, missing sandbox behavior, diagnostic path sanitization, and request bounds. The implementation constants and libproc structure layout were checked against the installed Apple SDK's `usr/include/libproc.h` and `usr/include/sys/proc_info.h`; the sandbox invocation and deprecation notice were checked against the installed `sandbox-exec(1)` manual.
