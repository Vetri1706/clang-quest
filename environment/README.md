# Local AI and C++ environment

There are two environments in this repository: the Python/NumPy process that trains and runs the small model, and the restricted C++ execution environment that checks learner programs. Neither requires Docker or a hosted service.

## Install the AI runtime

Use Python 3.11 or newer for NumPy 2.4.6. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ai.txt
source .venv/bin/activate
```

The pinned dependency is [requirements-ai.txt](../requirements-ai.txt). The dashboard's grounded guide and compiler otherwise use only Python's standard library. The measured development environment used CPython 3.14.6 and NumPy 2.4.6 on macOS arm64; the current verification report records the actual compiler and machine characteristics, without recording usernames or secrets.

Build the UI once with Node.js 22.13 or newer, then use the local environment to launch:

```sh
npm ci
npm run build
.venv/bin/python start.py
```

`Start Questline.command` prefers `.venv/bin/python` when present. It falls back to `python3` for an existing system setup. The installed interpreter, compiler tools, and virtual environment are not committed to Git. The model source, tokenizer, metadata, and small NumPy weight archives are committed.

## Inference entry points

Run the bundled DPO checkpoint directly:

```sh
.venv/bin/python ai/generate.py ai/runs/hardened_dpo 'What is RAII?' --tokens 24 --temperature 0
```

In the dashboard, select **Your NumPy model · experimental**. The API's `/api/mentor` operation dispatches to [backend/model_bridge.py](../backend/model_bridge.py), which loads the in-repository checkpoint and uses paged cached decoding in [ai/generate.py](../ai/generate.py). The default grounded guide uses [backend/mentor.py](../backend/mentor.py); it does not use neural weights.

The separate [ai/serve.py](../ai/serve.py) entry point is a bounded loopback inference service with a locally generated private bearer token. It is available for framework experiments, but the dashboard does not need that additional server. Token files are ignored by Git.

## The C++ environment

[backend/runner.py](../backend/runner.py) compiles real C++20 source with Apple Clang. Each request gets a temporary directory and fixed compiler arguments. macOS Seatbelt restricts private files, runtime file writes, network access, and process creation. CPU, time, output, and sampled RSS are bounded; a compile semaphore allows only one job at a time. There is no unrestricted fallback.

[backend/data/curriculum.json](../backend/data/curriculum.json) contains the original 24 missions, examples, hidden test cases, teaching hints, and reference programs. [backend/server.py](../backend/server.py) selects tests and removes private inputs/outputs before transport. [backend/store.py](../backend/store.py) awards XP transactionally after a first successful submission. The local database stays outside Git.

The application currently supports macOS for real C++ execution. Install missing Apple Command Line Tools with `xcode-select --install`. The Seatbelt command-line interface is deprecated, and memory checks are sampled rather than a hard kernel resident-memory quota. This environment is designed for a personal learning lab, not an Internet-facing multi-user judge.

## Is this an RL environment?

The compiler returns pass/fail outcomes that can be used as feedback, but this application does not perform online reinforcement-learning updates from learner submissions. The included training alignment is DPO on three fixed chosen/rejected pairs. It is not PPO, policy-gradient training, or an autonomous RL agent learning through repeated compiler interactions. See [the training basis](../ai/TRAINING_BASIS.md) for the actual learning objectives and data.

## Reproduce the environment test

```sh
.venv/bin/python scripts/verify_ai_environment.py --output reports/ai
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python ai/verify.py
.venv/bin/python ai/evaluate.py ai/runs/hardened_dpo --output reports/ai/cpp-quality.json
```

The environment verification launches an ephemeral real HTTP server and a temporary progress database. It invokes the actual model, obtains compiler and mentor feedback, and records generated output and checks in JSON and Markdown. It leaves your saved learning profile unchanged. Passing execution checks is separate from passing the C++ knowledge screen; the current checkpoint fails the latter.
