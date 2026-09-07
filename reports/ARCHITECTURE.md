# Questline implementation notes

Questline is a single-user local learning application. The shipped interface is statically exported React; a Python standard-library HTTP server serves those assets and a JSON API. The frontend never receives a shell, filesystem API, or unrestricted compiler endpoint. The application owns a fixed curriculum, and each compile request selects one mission and one of three execution modes.

## Main components

| Component | Responsibility |
| --- | --- |
| `app/page.tsx` | Shared navigation, mission selection, editor lifecycle, autosave, run/submit state, mentor messages, and preferences |
| `components/quest/code-editor.tsx` | CodeMirror C++ language support, keyboard execution, read-only run state, and cleanup |
| `components/quest/practice-panels.tsx` | Problem/concept tabs, test results, safe mentor text/code rendering, and reference links |
| `components/quest/learning-views.tsx` | Dashboard, difficulty/path/search filtering, path cards, and derived achievements |
| `backend/server.py` | Loopback HTTP boundary, request validation, concurrency gates, API orchestration, and static delivery |
| `backend/runner.py` | Compilation, restricted child execution, stream bounds, CPU/wall/RSS monitoring, and process-group cleanup |
| `backend/mentor.py` | Local retrieval, progressive hints, source review, compiler/test explanations, and adaptive teaching |
| `backend/model_bridge.py` | Lazy bounded generation through the bundled `ai/` NumPy framework |
| `ai/generate.py`, `ai/rawllm/` | Actual NumPy inference, model/autograd/cache/tokenizer, training and distributed implementation |
| `ai/runs/hardened_dpo/` | Active trained checkpoint, tokenizer, settings, and frozen DPO reference |
| `backend/store.py` | SQLite migrations, drafts, attempts, messages, activity, and unique completion awards |

The app uses matching shadcn sidebar, tabs, dialogs, progress, buttons, input, and select components. It uses system fonts and packaged assets. No remote scripts, CDN assets, hosted model endpoint, telemetry, or account service is required at runtime. Primary documentation links open only when the learner follows them.

## The run transaction

The editor submits JSON with its mission ID, source, and run mode. The server validates its loopback Host, Origin/fetch context, session CSRF token, JSON content type, body framing, string types, and byte limits. Unknown mission IDs are rejected. A nonblocking semaphore admits at most one compiler operation; busy requests receive HTTP 429.

`run` selects the two visible fixtures. `submit` selects all five. `custom` supplies one user-defined stdin string, reports execution, and never counts as a graded completion. C++ source is written into a fresh job directory. The compiler is invoked with fixed argument arrays, without shell interpolation. It produces one executable for all cases in the request.

A trusted helper installs resource restrictions before invoking macOS Seatbelt. Runtime execution cannot spawn more processes or write files. The parent uses nonblocking pipes, bounded captured output, wall deadlines, CPU limits, and libproc-based memory monitoring. It terminates the child process group and closes temporary resources even on runner errors. The HTTP server uses non-daemon request threads so normal shutdown drains each request through its cleanup path.

The API removes hidden inputs, expected values, stdout, stderr, and actual output before either persistence or browser transport. A safe zero-based `test_index` preserves the mentor's fixture mapping, while `index` is one-based for the learner. Custom runs avoid fixture fallback. Results include run mode and timing so the mentor can distinguish an output mismatch from compiler unavailability or a resource limit.

SQLite then records the attempt and draft in one transaction. A completion row is inserted only for an all-passing submission. The primary key on mission ID makes a second successful submission a no-op for XP, even if requests arrive concurrently through independent application instances. Activity totals and detailed attempts update in the same transaction. The UI applies the returned profile and opens the earned-XP dialog only when a positive award was actually committed.

## Persistence and navigation

Draft saves use a client-side promise queue. A short debounce avoids a write per keystroke. Mission changes flush outstanding edits before switching; edits are paused during mission fetches and compilation. A dedicated last-challenge setting is updated in the draft/attempt transaction, avoiding timestamp ties when several saves happen within a second.

SQLite uses bound parameters, WAL mode, synchronous FULL, and a five-second busy timeout. Each request opens and closes its own connection. Detailed attempts are bounded to 500 records; message history is bounded to 40 messages per mission. Completed mission totals and per-day aggregates persist independently of that detail retention. Private state is excluded from the release archive.

Progress is local and tied to this data directory. There is no sign-in or multi-user separation. Editing the database or reading the developer fixtures is within the local owner's control; this is not an exam anti-cheating system. Badges represent practice in this application, not externally certified competence.

## Mentor boundaries

The default guide retrieves original teaching cards with a small BM25 index and a topic-overlap gate. Known diagnostic patterns explain actual Clang evidence conservatively. Test feedback uses visible values only. Source review looks for specific lexical indicators and phrases findings as checks or possibilities, rather than claiming a complete semantic analysis.

The saved learning difficulty takes precedence over XP rank. A profile path of `all` falls back to the current challenge's path. Hints advance through a maximum of three stages. Solution disclosure requires a positive, explicit request; negated or unrelated requests do not trigger it. Unknown topics receive a bounded admission rather than an invented answer.

The optional experimental bridge resolves the NumPy framework under this repository's `ai/` directory and loads its own non-pickle checkpoint format. Before allocation, the dashboard rejects configurations above two million parameters or 512 context tokens. The shipped model has 29,656 parameters and a 256-token context. The framework validates checkpoint metadata and tensor data. A single model lock prevents simultaneous generation; output is short, thread counts are capped, and the prompt is shortened to fit a bounded output budget. Tiny or unusable context windows fail intentionally instead of looping forever. The UI labels this checkpoint as having failed its knowledge gate. Neural output never automatically executes code or determines XP. The [environment verification](ai/environment-validation.md) explicitly submits one captured generation to the compiler and records its failure.

## Delivery and trust limits

The production process binds to `127.0.0.1`; static file delivery resolves paths under `dist/client` and blocks traversal. API responses disable caching and MIME sniffing, set a frame-denial policy, and avoid dumping internal tracebacks. No third-party request origin receives permissive CORS headers. Oversized/malformed input receives a bounded error response. Eight request slots and one compiler slot limit concurrent work.

This is deliberately a local deployment. Apple's deprecated Seatbelt launcher, sampled RSS, finite runner regression tests, absence of public-service authentication, and a small curated mentor remain material boundaries. Sudden power loss or force-killing the entire Python process bypasses graceful application shutdown; it is different from the tested Control+C/SIGTERM drain path. Source fixtures should not be treated as secret against the local machine owner.

The GitHub repository includes the source, original curriculum, tests, reports, and self-contained AI weights and framework. The compiled frontend is generated locally using `npm ci` and `npm run build`; the repository excludes personal learning databases, virtual environments, and Node dependencies. No training runs in the background. Ordinary operation requires Python and the installed Apple compiler, plus NumPy for the experimental model. [Environment setup](../environment/README.md) records the pinned dependency and supported execution platform.
