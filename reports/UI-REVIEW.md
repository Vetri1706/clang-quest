# UI and API integration review

Reviewed `outputs/cpp-quest/app/page.tsx`, `components/quest/practice-panels.tsx`, `components/quest/code-editor.tsx`, and `components/quest/learning-views.tsx` against `lib/quest-api.ts`, `lib/quest-types.ts`, `backend/server.py`, and `backend/store.py`.

This is a read-only source review. No browser, DOM automation, visual QA, Sites operations, or checkout edits were performed. The sequences below follow the actual callbacks and API behavior; they are not claims of browser reproduction. Line numbers refer to the reviewed version and may shift as the owner applies fixes. All findings were sent directly to the owner.

## Final follow-up status

The owner applied the six fixes described below. Follow-up source inspection confirms synchronous transition guards and response identity checks, captured-source writes, actual-write acknowledgements plus debounce reconciliation, pending-save unload protection, disjoint message key namespaces, award display after the final write, settings-only versus progress-only profile merges, and serialized settings updates. Mode changes also now carry a revision and last-acknowledged persisted value, so stale responses cannot reset the current selection and the latest failed write rolls it back.

Result revisions now carry `source_sha256`; the backend supplies `source_changed` for differing or legacy unversioned results, and the console labels earlier-draft output. The copied mentor prefixes stale diagnostic/test/progress evidence with a clear notice, omits mismatched current-code excerpts, and preserves current-code review. All 35 focused mentor tests passed, including four new stale-evidence regression methods. The owner is running the combined suite; this review did not repeat it.

The final CodeMirror command-level finding was also corrected. Installed `@codemirror/commands/dist/index.js:287` checks `state.readOnly` before undo/redo, while the original editor set only `EditorView.editable`. The owner now configures both facets in the initial compartment and its reconfiguration, preventing editing commands during a mission transition. This was checked against the installed dependency source and the final editor source, without browser or DOM execution.

No unresolved concrete defect remains from this bounded review. This is source-level verification of the reviewed interactions, not a claim of exhaustive browser race testing or visual QA.

The historical findings below document the original triggers and corrections; they should not be read as still-open defects after the follow-up status above.

## P1 — A run started during mission loading can overwrite another mission's draft

Locations: `app/page.tsx:134` (`loadMission`), `app/page.tsx:235` (`run`), `components/quest/code-editor.tsx:45` (Mod-Enter callback).

`loadMission` checks the run/mentor refs only at entry, then waits for saving and a GET. It sets React `switching` state, but `run` only checks `busyRef`; `ask` only checks `mentorRef`. The CodeMirror keymap directly invokes the run callback without a read-only or switching guard. The registered `run_current_code` tool also calls the same callback.

Sequence: mission A is open; opening B starts its asynchronous GET; a run starts before that GET returns. The run captures challenge A. Loading B then replaces `detailRef` and `codeRef`. When the run completes, it places A's result into B's UI and calls `save(A, codeRef.current)`, which now contains B's source. This can overwrite the saved source for A. A mentor response started during the same gap can append A's conversation to B's visible chat.

Correction: use a synchronous mission-transition ref checked by all action entry points; capture source and challenge identity together; refuse to apply any response to a different active mission. Save the captured source under its captured challenge rather than reading a mutable current source after an await.

## P1 — Undo during an in-flight autosave can silently persist the wrong source

Locations: `app/page.tsx:201` (autosave effect), `app/page.tsx:226` (before-unload guard).

Sequence: source A is loaded and `saved.current.source` is A. Edit it to B and wait until the 650 ms timer starts POST B. Before that response arrives, undo back to A. The effect now exits because saved A equals editor A. B's completion also skips the acknowledgement because the current editor differs from B. The database contains B, the editor contains A, and the saved ref incorrectly still contains A. Reloading replaces A with B without triggering the unsaved-change warning.

Correction: record what each completed write actually persisted, independently of whether that text is still visible; if current source differs after the write, explicitly queue the newest source or change a state dependency that schedules it. Merely changing a ref does not rerun the existing effect. The unload guard must account for pending writes and acknowledged database content.

## P2 — Chat messages acquire duplicate React keys after the second ask

Locations: `app/page.tsx:283` (append user and assistant), `components/quest/practice-panels.tsx:408` (`key={m.id || i}`).

Locally appended user messages have no database id, so they use their array index. Assistant replies carry their SQLite id. Starting with no messages, the first assistant has id 2. The next user message is at array index 2. Both therefore use key 2 in the same list; the pattern repeats on later messages. React reconciliation can reuse the wrong bubble, and the identities also change when slicing the list to 40 items.

Correction: provide every temporary message a stable client id and use a disjoint namespace for temporary versus persisted ids. Do not mix array indices and database ids in the same key space.

## P2 — The completion dialog can offer navigation while navigation is still blocked

Locations: `app/page.tsx:253` (set award before final save), `app/page.tsx:906` (Keep the momentum).

A successful run opens the award dialog before awaiting the final draft write. During that write, `busyRef` is still true. The enabled Keep the momentum button first clears the award and then calls `loadMission`, which refuses because the run is busy. With a delayed local request, the explicit navigation action closes the dialog and leaves the learner on the old mission with a notice.

Correction: open the award after the final save and busy-state cleanup, or disable its navigation action until the run has fully completed.

## P2 — Out-of-order profile snapshots can hide newly earned progress

Locations: `app/page.tsx:252` (run profile replacement), `app/page.tsx:308` (mode-setting response), `app/page.tsx:874` (preferences response), `backend/store.py:110` (returned profile snapshot).

Run results and settings responses independently replace the whole `data.profile`. The settings controls remain accessible while a run is active. A settings response can contain a profile read before the completion transaction, arrive after the successful run response, and replace the new XP/completions with its older snapshot. SQLite retains the completion, but badges, totals, next-mission selection, and the award's total XP can appear to go backward until another refresh. Overlapping setting updates can similarly replace each other's displayed values.

Correction: merge only the setting fields acknowledged by a settings request into the current profile, and preserve those settings when applying older run snapshots; alternatively use a shared mutation queue or server profile revision that rejects stale snapshots. A request-start sequence alone is insufficient if server processing order differs.

## P2 — Diagnostic feedback can quote a different revision as the reported source line

Locations: `app/page.tsx:277` (mentor request sends current editor source), `backend/server.py:108` and `backend/server.py:118` (last stored run plus current source), `backend/store.py:105` (attempt payload), `backend/mentor.py` (`_diagnostic_reply`).

After compiling source A, the learner may edit the code to B and click Help me understand this result. The stored run contains A's diagnostic, but it has no source hash or source snapshot. The mentor receives that diagnostic alongside B and extracts its line number from B, then labels it as the reported source line. The console similarly keeps old results beside new source without a stale-revision label. This can point the learner at a line that was never compiled.

Correction: associate each result with the submitted source revision. When the current source differs, label the result as belonging to the previous revision and either use the saved source snapshot for excerpts or omit the excerpt with a clear instruction to run the changed source.

## Confirmed matching contracts

- Public challenge fields used by the problem and learning views match the backend catalog schema. The UI never expects the hidden inputs or reference solution in challenge detail.
- The console handles visible and hidden results separately and only displays private-case status metadata. Its one-based display index agrees with the backend; the new zero-based test_index is mentor metadata and need not be rendered.
- Custom runs do not receive an expected-output field. The console tests for field presence and labels successful custom execution as finished rather than a verified solution.
- Earned XP, completed mission ids, daily counts, and achievement thresholds derive from actual stored completions rather than local sample-run success. The identified profile race affects the displayed snapshot, not the backend's transactional deduplication.
- Experimental model selection explicitly discloses its failed C++ quality gate. Mentor text is rendered as React text/code nodes rather than injected HTML.
- The backend remains the authority for request byte limits. Character limits in the textareas can still allow a multibyte string that the backend rejects; the returned error is shown rather than silently truncating it.
