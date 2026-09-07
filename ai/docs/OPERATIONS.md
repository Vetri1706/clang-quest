# Local operations and acceptance record

This document defines the supported deployment boundary: a trusted local POSIX
machine running bounded NumPy workloads. The supplied checkpoints are tiny
experiments, and the HTTP server is an operator tool for those checkpoints.
System correctness and model knowledge are evaluated separately. Docker,
external model services, cloud jobs, and background training schedules are absent.

## Verified behavior and outstanding deployment work

| Area | Implemented and exercised | Remaining boundary |
| --- | --- | --- |
| Tensor mathematics | Analytical backward, finite differences, dense/tiled attention parity | Native accelerator kernels and large-shape performance |
| 3D execution | Real MLA decoder, two data × two pipeline × two tensor ranks | Arbitrary planner, overlap, combined 3D/ZeRO schedule |
| Automatic ZeRO | Streamed initialization, operation rematerialization, tied embeddings, uneven shards | Activation checkpointing and large-machine benchmarks |
| Distributed recovery | Immutable shards, collective validation, atomic manifest, fresh-worker continuation | Changed membership, object stores, elastic recovery |
| Local training | Pretraining/SFT/DPO, exact resume, bounded flags, signals, run ownership | Indexed data seeking and automatic retention |
| Serving | Loopback, bearer auth, fixed workers, queue/cache/request limits, cancellation | Public deployment, TLS termination, sustained load study |
| Artifact parsing | Bounded JSON/ZIP/NPY, checksums, dtype/schema validation, no pickle | Authenticated external provenance and hostile parent directories |
| C++ evaluation | Original held-out pairs, stratified results, bootstrap intervals, explicit gate | Free-form teaching quality, complete curriculum, learner studies |

The source and tests support these concrete claims. Passing them does not certify
a system for an unmeasured deployment. The recorded small-model C++ evaluation
fails its paired-likelihood gate. The model must therefore not be presented as a
reliable C++ instructor, LLVM expert, or game-development assistant.

## Reproduce a bounded three-phase run

Use a fresh output directory for each new phase and each watchdog report:

```sh
python3 supervise.py --output results/local-pretrain --seconds 90 train pretrain --output runs/local-pretrain --steps 3
python3 supervise.py --output results/local-sft --seconds 90 train sft --initialize runs/local-pretrain --output runs/local-sft --steps 3
python3 supervise.py --output results/local-dpo --seconds 90 train dpo --initialize runs/local-sft --output runs/local-dpo --steps 3
python3 evaluate.py runs/local-dpo --output results/local-cpp-eval.json --require-pass
```

The last command writes its report before returning a nonzero status when the
likelihood gate fails. It does not alter model weights. A gate failure remains a
reported result; it is not hidden by removing difficult examples or changing the
threshold after looking at scores. Evaluation source material remains outside
the optimizer input fixtures.

The supervisor records `process.log` and `result.json`. Its result identifies
normal completion, child failure, deadline, output exhaustion, or cancellation.
It invokes an argument vector directly, without a shell, and starts a separate
POSIX process group. It forces one BLAS thread and lowers process priority. Its
output cap applies to captured stdout/stderr, not every file a child might write.
Its wall-clock limit is independent of the trainer's between-attempt deadline.

The supervisor sends SIGTERM, allows a finite grace interval, then uses SIGKILL
if necessary. It cleans descendants that retain stdout after the parent exits.
This is intended for trusted framework programs. A process deliberately escaping
its group is outside the contract. There is no operating-system resident-memory
limit; configured array limits and small defaults provide admission controls,
while Python, BLAS, allocator, and filesystem overhead remain real costs.

## Local stop, restart, and log consistency

`train.py` owns its output directory through `.run.lock`. The kernel releases
ownership when a process exits or dies. The file is intentionally retained:
unlinking and recreating it could produce two independently locked inodes and
allow concurrent writers. The metadata inside is diagnostic, not evidence that
a stale process remains alive. A second cooperating writer fails before writing
a run status or changing a checkpoint.

SIGINT and SIGTERM set a flag. The handler performs no file I/O and never updates
weights. The trainer completes the current optimizer boundary, commits its
checkpoint, and stops before the next attempt. `status.json` distinguishes
completed work, a requested stop, budget exhaustion, and failure. A native NumPy
kernel cannot be safely interrupted by a Python signal handler in its middle;
use the process watchdog when a hard time ceiling is needed.

```sh
python3 train.py pretrain --output runs/local-pretrain --resume --steps 5
```

The step argument is the desired total number of attempts for the local trainer.
Resume restores moments, master weights, random state, loss scale, counters, and
deterministic sample position. Source hash, tokenizer hash, objective, precision,
sequence length, accumulation, and preference beta must agree. A changed
experiment initializes a new output rather than rewriting a previous history.

Checkpoints are authoritative. Telemetry rows are appended after the manifest
commit and fsynced. If termination occurs between commit and log append, a final
row can be absent even though its update is recoverable. Resume does not invent
that observation. It drops uncommitted or truncated tail records, rejects
corruption before later committed records, and resumes from checkpoint counters.
Data replay uses bounded memory and checks stopping conditions, but it remains
linear in samples already consumed. There is no indexed seek table.

Old immutable payloads remain on disk. Capacity planning must include retained
generations and abandoned preparation files. For manual archival, first stop
writers and readers, preserve every payload referenced by a retained manifest,
and move complete run directories together. The project does not silently prune
by modification time or automatically delete a potentially useful checkpoint.

## Distributed sharded restart

The automatic ZeRO arithmetic example can commit and restore actual optimizer
shards. Each process owns only its partition of parameters, accumulated
gradients, first moments, and second moments. Full parameters are temporary
operation inputs, and the graph retains activation inputs instead of gathered
weight arrays.

```sh
python3 zero_train.py --checkpoint results/local-zero-state --output results/local-zero.json
python3 zero_train.py --resume results/local-zero-state --steps 1 --output results/local-zero-resume.json
```

Here `--steps 1` means one additional update after restore. This example compares
the result against a dense reference, so it retains small verification traces
whose extra bytes are explicitly reported. These traces are not intrinsic
optimizer storage. The numerical fixture is deterministic token data and does
not establish language learning.

The checkpoint directory must be visible to every participating rank. The
coordinator does not transport checkpoint files. Each rank writes and fsyncs an
immutable NPZ payload and metadata file. After collective preparation, the
leader revalidates every shard and atomically replaces the manifest. A worker
dying before this point leaves the previous committed generation intact.
Unreferenced files are ignored on restore. A connection failure after the rename
can leave the caller uncertain about save success; inspect the manifest to
resolve the generation before retrying.

Loading validates destination shape and writability, incoming layout and
membership, checksums, numerical finiteness, nonnegative variance, counters,
random-generator state, and run ownership before collective agreement permits
mutation. Tests launch entirely new processes and a new TCP session and compare
continued arrays and RNG values byte for byte. Different world sizes, topology
resharding, and failed-coordinator consensus are not supported. The advisory
lock and rename contract is for a shared local POSIX filesystem, not an assumed
property of every remote filesystem.

For actual text and SFT training, use the corpus entry point:

```sh
python3 supervise.py --output results/zero-corpus-job --seconds 60 zero_corpus pretrain --output runs/local-zero-corpus --steps 2
python3 zero_corpus.py pretrain --output runs/local-zero-corpus --resume --steps 3
python3 zero_corpus.py sft --output runs/local-zero-sft --steps 2 --report results/local-zero-sft.json
```

These steps are total completed updates, unlike the additional-update option
in the numerical `zero_train.py` probe. `--data` supplies a `.txt` or JSONL source;
SFT requires JSONL prompt/response fields. The CLI stages a hashed copy with a
16 MiB byte limit and trains its own tokenizer using at most 64 KiB. It uses a
small one-block model with the actual tokenizer vocabulary. Rank processes
replay the same global stream and select disjoint slots. A loss numerator is
multiplied by data-degree/global-supervised-token-count before mean
reduce-scatter, giving the correct global token objective for unequal masks.

Checkpoint counters include global replay position, consumed target counts,
settings/config/data/tokenizer hashes, and rank ownership. Resume requires a
committed manifest and unchanged metadata; a missing manifest is an error,
rather than permission to restart silently at step zero. SIGTERM unwinds the
parent's worker-management context and leaves the last committed generation
available. Exact continuation and unequal-token SFT parity are tested. This path
does not provide DPO or a dense serving-format export; use the local three-phase
trainer when that format is needed.

## Local inference service

```sh
python3 serve.py runs/hardened_dpo --workers 1 --queue 1 --port 8080 --cache-mib 8
```

The service accepts loopback addresses only. The default bearer secret is
generated into `RUN/serve.token` with mode 0600; existing token files must be
regular, nonsymlinked, owned by the current user, and inaccessible to other users.
FIFO paths are rejected without waiting on a writer. Alternatively, provide
`RAWLLM_API_TOKEN` through the process environment. Secrets are never printed,
embedded in result reports, or included by the release builder.

This standard-library client reads the token locally and sends it in the HTTP
header without placing it in shell command arguments:

```python
import http.client
import json
from pathlib import Path

token = Path("runs/hardened_dpo/serve.token").read_text().strip()
connection = http.client.HTTPConnection("127.0.0.1", 8080, timeout=10)
payload = json.dumps({"prompt": "C++", "max_tokens": 16,
                      "temperature": 0, "seed": 5})
connection.request("POST", "/generate", payload,
                   {"Authorization": "Bearer " + token,
                    "Content-Type": "application/json"})
response = connection.getresponse()
print(response.status, json.loads(response.read()))
connection.close()
```

`GET /health` and `GET /metrics` expose operational state without prompts,
weights, filesystem paths, or credentials. `POST /generate` requires bearer
authentication. Unknown request fields, invalid scalar types, invalid Unicode,
oversized prompts, and context overflows produce safe JSON errors. The server
does not silently truncate a prompt. Generation has a request-local RNG, so a
seed does not depend on other requests' sampling order.

The HTTP implementation uses one request per connection. It rejects chunked or
ambiguous framing, conflicting length headers, unexpected Host values, and
oversized headers or bodies. A fixed worker pool and bounded admission slots
control executing and queued connections. Idle clients have I/O deadlines.
Accepted queue time contributes to request duration, and saturated capacity is
observable through counters and rejection responses.

Each generation owns an independent paged cache. A shared accounting lock
reserves its complete array footprint before allocation, including reference
arrays. Cache accounting is returned to zero through cleanup on success,
exception, timeout, disconnect, or shutdown. The cache limit excludes model
parameter arrays, Python containers, allocator overhead, request parsing, and
temporary matrix operations. Those exclusions are in the evidence report.

Shutdown stops admission, interrupts idle sockets, cancels active generation at
decode boundaries, and joins the bounded workers. Existing arithmetic cannot be
safely preempted in the middle of a NumPy kernel. The measured service run loaded
the actual saved model, matched standalone paged decoding, returned authenticated
HTTP 200, reclaimed cache arrays, and exited cleanly on SIGTERM. It was a small
functional test, not a sustained throughput or concurrency benchmark. Stop the
service when finished; no service remains running in the delivered workspace.

`python3 load_serving.py` starts a separate temporary local server, sends a
bounded 64-request probe, compares successful responses with a deterministic
standalone reference, checks final cache accounting, and stops the server. Its
temporary credentials are removed. The recorded run had 58 successes, six clean
capacity rejections, no mismatches, no errors, and zero residual cache bytes.
The success latency percentiles were 8.97 ms, 11.27 ms, and 14.07 ms at the 50th,
95th, and 99th percentiles. These observations cover a 0.139-second tiny workload;
they are not a sustained benchmark, service-level guarantee, or large-model
performance prediction. Regenerating the report can change scheduling-dependent
counts and latency, while the correctness checks remain required.

## Memory, parsing, and artifact integrity

For N owned float32 parameters, automatic ZeRO's persistent parameter, gradient,
and two Adam arrays use exactly 16N bytes. The 8,556-parameter, two-rank fixture
owns 4,278 elements per rank, or 68,448 bytes, before object overhead. A forward
operator can additionally materialize its full weight. Backward can require that
weight, one full gradient, input gradients, retained activations, network copies,
and local optimizer candidates. Initialization temporarily holds NumPy's FP64
random output before casting. Reported shard ownership is not total peak RSS.

Local AdamW retains a full model, gradients, master weights, and two moments,
approximately 20 bytes per float32 parameter. Transactional candidates add
approximately 16 bytes per active float32 parameter plus temporary arithmetic.
This extra storage prevents a finite-gradient overflow from producing partial
updates. The large preset is analytical metadata: one complete float32 weight
copy alone exceeds 28 billion bytes. It is never allocated by the supplied
verification or training commands.

NPZ is a ZIP archive of NPY arrays; NPY headers describe shape and dtype. The
reader therefore validates the ZIP end record and central-directory bounds
before constructing archive metadata, then validates every declared NPY header
length before allowing NumPy to read it. Inventory, decompressed extent, and
expected dtypes are checked before any array is reconstructed. These checks
build on the documented [NumPy binary format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
Hashes are computed against the same open regular file used for reading. JSON
rejects duplicate fields, nonfinite constants, and exponents that overflow a
Python float. BPE merge expansion has token, total-byte, and merge-count limits.

Parent directories and concurrent local filesystem writers remain trusted.
Checksums do not authenticate a hostile replacement manifest. Kernel locks
coordinate cooperating POSIX processes through [fcntl](https://docs.python.org/3/library/fcntl.html).
The watchdog uses the documented argument-vector and process-session facilities
of [subprocess](https://docs.python.org/3/library/subprocess.html). None of these
mechanisms makes generated code safe to execute; the evaluator runs no generated
C++ programs and invokes no compiler on model output.

## Release procedure

```sh
python3 verify.py
python3 release.py
python3 release.py --check
```

`verify.py` executes numerical, process, recovery, parsing, serving, and
evaluation tests and audits source imports for the prohibited ML frameworks.
`release.py` records source/evidence SHA256 values and tests the resulting ZIP.
Runtime locks, token files, private keys, caches, and intermediate partial files
are excluded. The archive contains complete source and small measured artifacts.
Rebuilding after a source edit intentionally changes its manifest. Passing
integrity verification proves consistency with the included hashes, not the
identity of an external publisher or production suitability.
