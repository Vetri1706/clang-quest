# Questline — your local C++ practice lab

A gamified learning workspace with an embedded C++ editor, real compilation and tests, a local study mentor, and persistent progress. It runs entirely on your Mac. No Docker, accounts, API keys, or paid services are required.

## Clone from GitHub

The repository contains source code. Build the frontend once after cloning:

```sh
gh repo clone Vetri1706/clang-quest
cd clang-quest
npm ci
npm run build
python3 start.py
```

This setup requires Node.js 22.13 or newer. Once built, ordinary use only needs Python 3.10 or newer and Apple's Command Line Tools. The local release ZIP already includes the compiled interface and can skip the npm steps.

## Open the app

Double-click **Start Questline.command**, then open **http://127.0.0.1:5173**. Keep the terminal window open while you practice. Press Control+C there to stop the app; active compiler requests drain before shutdown.

Alternatively, from this folder:

```sh
python3 start.py
```

After a source build, or when using the local release ZIP, the compiled interface lives in `dist/client`. Ordinary use then does not need Node.js or an npm development server. Python 3.10 or newer and Apple's Command Line Tools are required. If the restricted compiler is unavailable, the dashboard explains the engine status. Install missing Apple tools with `xcode-select --install`, then restart Questline. The runtime does not fall back to unrestricted execution.

Use `python3 start.py --port 5174` if another application already uses the default port. The app listens only on the loopback interface. It is a personal local application, not an Internet hosting service.

## Practice, then make it yours

1. Open **Practice arena** or pick a challenge in **Real-world labs**. Filter by beginner, intermediate, or advanced level, or select a learning path.
2. Read the scenario and examples. The **Learn the concept** tab explains the relevant C++ ideas.
3. Edit `main.cpp`. The CodeMirror editor includes syntax highlighting, line numbers, indentation, bracket matching, search, and undo. Drafts save after a short pause. Command+Enter or Control+Enter runs the examples. The editor pauses changes during a run or mission switch.
4. **Run code** compiles with Apple Clang in C++20 mode and checks the two visible examples. **Custom input** runs your own stdin without grading it.
5. **Submit** checks all five cases, including three hidden cases. Passing every case awards the mission's XP once. Rerunning a completed solution cannot farm XP.
6. Ask the mentor for a hint, a concept explanation, a review of your approach, or help interpreting the latest compiler/test result. Hints progress through three stages. A complete reference solution requires an explicit request such as “show solution.”

The sidebar profile control changes your name, preferred difficulty, learning path, and daily goal. The dashboard shows actual activity, XP, streak, path completion, and earned badges. A streak counts calendar days with a practice attempt; the daily goal counts newly completed missions. There are no simulated users or leaderboard scores.

## 24 original missions

Eight beginner, eight intermediate, and eight advanced problems span four paths:

| Path | What you practice |
| --- | --- |
| Foundations | Input/output, integer arithmetic, branches, loops, strings, and collections |
| Systems & memory | Queues, allocation models, caches, scheduling, and resource reasoning |
| Game development | Movement, collision math, grids, inventories, and pathfinding |
| Compiler workshop | Tokenization, expression evaluation, symbol tracking, and dependency algorithms |

Problems use realistic scenarios and deterministic text input/output. Game missions teach engine concepts through console programs; this release does not embed a graphical game engine. Compiler missions teach relevant algorithms and mentor references include LLVM; it does not build LLVM inside a browser tab.

Each mission includes starter code, objectives, concepts, three hints, and five tests. All 24 reference programs passed all 120 cases during curriculum validation; all 24 starter programs also compiled. The content is original and does not depend on downloading third-party datasets. See `backend/data/CATALOG.md` for the full catalog.

## How the mentor works

**Grounded study guide** is the default. It uses an original local curriculum, 24 reference cards, deterministic retrieval, conservative code checks, and actual compiler/test evidence. It adapts to the difficulty you choose and the current mission's domain, and links to primary documentation. It runs with the Python standard library and sends no requests to an external AI service. This is a bounded teaching assistant, not a newly trained general-purpose LLM. It admits when a topic is outside its reference set.

**Your NumPy model · experimental** connects to the custom model created earlier in the adjacent `numpy-llm-framework` project. That 29,656-parameter checkpoint has not passed its C++ knowledge gate. It is available for testing actual local inference, not as a verified source of programming explanations. It produces a short, clearly labeled continuation. Switch back to the grounded guide for learning help.

The experimental option requires NumPy and this layout:

```text
workspace/
  clang-quest/  # cpp-quest in the local release ZIP
  numpy-llm-framework/
    generate.py
    runs/hardened_dpo/
```

If you clone this repository or move/share the dashboard alone, the grounded mentor and compiler remain usable, while the experimental option becomes unavailable. The new app imports no PyTorch, TensorFlow, JAX, Hugging Face, or LangChain code. No training job runs in the background.

## Your saved work

Drafts, attempts, conversations, hint progression, preferences, and completions live in `state/progress.sqlite3`. The server uses SQLite transactions, WAL journaling, and a unique completion key for each mission. It retains the latest 500 detailed attempts and 40 messages per mission, while keeping aggregate activity and completion totals.

To back up your progress, stop Questline and copy the entire `state` folder. The distributed ZIP excludes personal progress, caches, dependencies, and temporary compiler jobs. Source and private test fixtures are included in the developer project because you own the local application, but the HTTP server exposes only compiled frontend assets and its bounded API. Hidden test inputs, expected values, stdout, and stderr are omitted from run responses and mentor feedback.

## Local execution limits

Only one compiler job runs at a time. Runtime code is restricted with macOS Seatbelt: private-file reads, filesystem writes, outbound connections, fork/spawn, and other-process information/signals are denied. Compiler reads are limited to its job directory and required toolchain/system files. Environment variables are cleared except for a minimal fixed environment; stdout and stderr are bounded and scrubbed of temporary paths.

| Resource | Limit |
| --- | --- |
| Source | 32 KiB |
| Input per case | 16 KiB |
| stdout / stderr per case | 16 KiB each |
| Compilation | 15 seconds; monitored 512 MiB RSS |
| User program | 2 seconds; monitored 256 MiB RSS |
| macOS binary startup | Separate maximum of 10 seconds per case |
| Simultaneous compilation | 1 |
| Concurrent HTTP requests | 8 |

macOS can spend time validating a newly compiled executable before user code begins. The runner reports startup separately and applies CPU and memory monitoring throughout. RSS is sampled every 20 ms, so brief overshoot is possible. Seatbelt's command-line interface is deprecated by Apple. These constraints are useful for a personal learning lab; they are not a certification for running arbitrary hostile programs as a public service. See `reports/RUNNER.md` for the implementation boundary.

## Develop and verify

Node.js 22.13 or newer is used only for frontend development. This repository retains the Sites/Vinext starter and its component catalog, with matching shadcn controls and CodeMirror added for editing. The production entry point is the Python launcher, not Wrangler or a Node SSR server.

```sh
npm ci
npm run typecheck
npm run lint
npm run build
python3 -m unittest discover -s tests -v
python3 start.py
```

For live frontend development, run these in separate terminals:

```sh
python3 -m backend.server --port 8766
npm run dev
```

Vite binds to `127.0.0.1:5173` and proxies `/api` to port 8766. Stop the normal Python launcher first to free port 5173. `npm run lint` checks authored app/components/API/configuration code; the unchanged starter component catalog is outside that lint scope. TypeScript still checks the project.

The test suite covers real compile/run failures and success, compiler restrictions, resource limits, transactional XP, persistence, malformed requests, CSRF/origin/host checks, hidden-data handling, mentor adaptation, actual compiler-feedback integration, and graceful request draining. Tests use temporary databases and do not change your learning progress.

Feature-detected WebMCP tools expose reading learning state, opening a mission, and running visible example tests through the same application actions. They remain optional; unsupported browsers retain every visible control. No supported browser WebMCP validation context was available during delivery, and browser interaction/visual QA was not performed. See `reports/VERIFICATION.md` for measured checks and remaining limits.
