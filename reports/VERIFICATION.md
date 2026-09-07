# Release verification — 7 September 2026

The local dashboard is served at **http://127.0.0.1:5173** by the packaged Python launcher. The Vite development server and separate development API process were stopped. No training process, Docker service, or remote AI service was started for the dashboard.

| Check | Result | Evidence |
| --- | --- | --- |
| Combined Python tests | **77 passed**, zero skips, 40.176 seconds | `backend-tests.txt` |
| Mentor coverage | 35 tests in the combined suite | `tests/test_mentor.py` |
| HTTP and persistence coverage | 25 tests in the combined suite | `tests/test_backend.py` |
| Restricted compiler coverage | 11 tests in the combined suite | `tests/test_runner.py` |
| Real API/compiler journey and lifecycle | 6 tests in the combined suite | `tests/test_integration.py` |
| TypeScript | Passed, no diagnostics | `typecheck.txt` |
| Lint on authored application code | Passed, no diagnostics | `lint.txt` |
| Static frontend build | Passed with Vinext/Vite | `frontend-build.txt` |
| npm dependency audit | **0 known vulnerabilities** | `npm-audit.json` |
| Production launcher HTTP checks | Root, API bootstrap, and all 15 static files returned 200; bytes matched disk | `local-launch.json` |
| Curriculum validation | 24 reference solutions, all 120 cases passed; all 24 starters compiled | `curriculum_validation.json` |
| Actual NumPy inference | Passed, 1.303 seconds; experimental warning present | `model-inference.json` |
| UI/API source review | Seven identified issues addressed | `UI-REVIEW.md` |

The real integration journey compiled a failing starter, obtained a progressive hint, submitted a reference solution through HTTP, passed all five cases, awarded XP, and restored source and conversation from a reopened SQLite store. Separate requests executed custom input and explained an actual Clang diagnostic. A later-draft regression verifies that the mentor does not attribute old compiler evidence to newly edited source.

Concurrency tests verify that eight simultaneous completion transactions produce only one XP award, while overlapping compiler requests are rejected until the admitted request releases the gate. Shutdown verification confirms that server close waits for an active request's cleanup. The runner tests cover private-file access, network connection attempts, process spawning/signals/information, writes, infinite loops, stdout/stderr floods, memory exhaustion, missing sandbox support, and unavailable accounting.

A combined run during frontend compilation exposed an overly short three-second deadline in a synthetic concurrency fixture. Its HTTP/test coordination deadlines were increased to 10–15 seconds; the actual compiler and program limits were unchanged. The final 77-test run passed. Testing used isolated databases and did not award XP in the user's real profile.

The frontend review corrected autosave/undo ordering, commands entering during mission switches, ambiguous chat keys, an award appearing before the final save, stale settings overwriting progress, old results being attributed to current source, and CodeMirror undo bypassing the visible edit lock. Model-mode saves are serialized and failed changes restore the last acknowledged selection.

Dependency fixes were applied as compatible version groups with their lockfile, without force resolution or bypassing install-script policy. The final audit found no known advisories. This is a point-in-time dependency result, not a guarantee against undiscovered vulnerabilities. Build output reports a chunk larger than 500 kB; the complete static delivery is approximately 1.45 MB and is served locally.

## Verification limits

Browser interaction, screenshots, viewport resizing, and visual QA were not performed. Responsive layout, semantics, safe text rendering, and shared action wiring were checked at source level, with TypeScript and a production build. Feature-detected WebMCP tools are implemented but were not verified in a supported browser context.

The C++ engine is a restricted local macOS runner, not a hosted multi-tenant sandbox certification. Seatbelt tooling is deprecated, and memory limits rely on sampled RSS. The 24 original missions and curated mentor do not cover every programming topic. The adjacent 29,656-parameter NumPy checkpoint remains below the C++ knowledge quality gate and is explicitly experimental. See `RUNNER.md` and `ARCHITECTURE.md` for these boundaries.
