# Bundled AI verification — 7 September 2026

The actual NumPy checkpoint was loaded and invoked through the local application's HTTP API, and its output was passed unchanged to the real restricted C++ compiler. The inference and execution integration works. **The neural checkpoint fails its C++ quality screen and did not generate compilable C++ in the program probe.**

## Results and scope

| Measurement | Result | Full evidence |
| --- | --- | --- |
| Framework verification | 181 tests passed, 0 errors/failures/skips; 33.283 seconds | [Console log](framework-tests.txt), [JSON](framework-tests.json) |
| Application verification | 81 tests passed, 0 skips; 35.230 seconds | [Verbose console log](application-tests.txt) |
| Real AI/API/compiler journey | 36 of 36 execution checks passed; 8.526 seconds | [Readable report](environment-validation.md), [JSON](environment-validation.json) |
| Independent Git source export with pinned virtual environment | The same 36 checks passed; 19.786 seconds; no adjacent AI project | [Readable report](source-export/environment-validation.md), [JSON](source-export/environment-validation.json) |
| Actual neural outputs | Two prompts produced 24 dots; the LLVM prompt produced 24 question marks | [Prompts and outputs](environment-validation.md#observed-neural-outputs) |
| Neural program submitted unchanged | Real compiler returned `compile_error`; 0/2 visible tests passed | [Compiler result](environment-validation.md#neural-program-submission) |
| Grounded mentor's requested reference | All 5/5 cases passed; three hidden payloads redacted | [Learning journey](environment-validation.md#grounded-mentoring-and-a-real-learning-journey) |
| Paired C++ knowledge screen | **Failed**, 7/18 mean-token and 6/18 summed-token correct-answer preferences; all 18 records scored | [Full quality JSON](cpp-quality.json), [interpretation](../../ai/MODEL_CARD.md) |

The 36 checks include real checkpoint loading, three neural prompts, a deterministic greedy repeat, verbatim neural-output compilation, an actual compiler diagnostic explained by the grounded guide, hint/explicit-solution behavior, restricted reference-program execution, XP persistence, hidden-data redaction, unchanged neural weights, observed sandbox helper launches, and clean server shutdown. They are distinct from the 81 application tests and do not count a compiler rejection as a neural success.

The paired-answer screen compares provided correct and incorrect completions using response-token log likelihood. It is a small diagnostic dataset, not a general coding benchmark. Its pass condition requires both mean and summed preference accuracies of at least 75%, with both bootstrap lower bounds above 50%. The detailed per-example scores and intervals remain in the JSON; an evaluation process exiting successfully does not mean this quality gate passed.

## Environment and artifact identity

The current reports use CPython 3.14.6, NumPy 2.4.6, Apple M3 with 8 GiB RAM, macOS 26.5.1 arm64, and Apple Clang 21.0.0. The verification caps BLAS thread counts at one and uses a fresh temporary SQLite database and ephemeral loopback HTTP port. No Docker, external model service, GPU, or learner profile data is used. Static assets and browser interactions are outside this API/AI verification; the unchanged frontend's earlier checks are in [the dashboard report](../VERIFICATION.md).

The source-export check materialized all 760 then-staged repository files into a new temporary directory using `git checkout-index`. That directory had no sibling `numpy-llm-framework`, user progress, virtual environment, or Node dependencies. Its copied verification script loaded its own `ai/` and backend sources, using the separately created repository `.venv` containing the pinned NumPy installation. This checks source/artifact portability on the same Mac; it is not a second-platform test or a frontend build. Its 36 checks repeat the same journey and are not counted as 36 additional distinct tests. Both environment reports record matching runtime-source, script, and checkpoint hashes.

The active 29,656-parameter DPO payload is:

```text
ai/runs/hardened_dpo/checkpoint/weights-09437d05fd214570a2786922d365d1d1.npz
SHA-256 f803d1cf276ae9bb02858d37f516a2c3aa7a455d9213d03ffe62318bd4e51bf5
```

The environment report includes this archive hash, a hash of the actual loaded numerical weights, the checkpoint/settings/tokenizer hashes, model configuration, generation flags, runtime limits, verification script hash, and hashes of the relevant runtime sources. Framework verification separately hashes its 44 audited Python sources and records NumPy as the only statically imported third-party Python package. [The training basis](../../ai/TRAINING_BASIS.md) records each stage's data, update counts, and lineage.

Bundled training artifacts were not retrained during integration. Framework tests create their own small temporary training experiments. The environment journey performs inference and compiler interaction without updating model weights; it is not an online RL run.

## Reproduce from the repository root

After following [the environment setup](../../environment/README.md):

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python ai/verify.py
.venv/bin/python scripts/verify_ai_environment.py --output reports/ai
.venv/bin/python ai/evaluate.py ai/runs/hardened_dpo --output reports/ai/cpp-quality.json --max-seconds 30
```

The [environment verification script](../../scripts/verify_ai_environment.py) returns zero only when its mechanical checks pass. Read the separately reported neural observations and quality gate. These commands write current reports or temporary test data; elapsed times can differ by machine and load. `ai/verify.py` writes `ai/results/verification.json`; the committed `framework-tests.json` is a copy from the recorded run. `application-tests.txt` captures the verbose unittest output.

Reports under [ai/evidence/historical](../../ai/evidence/historical/) predate relocation. Their metadata location normalization is recorded separately, and some refer to older experiments that are not bundled. They must not be substituted for the current checkpoint's evaluation.
