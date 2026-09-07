# Bundled NumPy language-model framework

This directory contains the actual inference and training implementation used by Questline's experimental model, together with its trained checkpoint, tokenizer, original training fixtures, and mathematical documentation. It is self-contained inside this repository and no longer depends on a sibling project.

The shipped model has **29,656 parameters**. It is a small software-validation experiment initialized from scratch, with three pretraining updates, three SFT updates, and three DPO updates. It is not a trained 7B model and is not ready to teach C++ reliably. The default dashboard mentor is the separate grounded guide.

## Find the implementation and artifacts

| Item | Location |
| --- | --- |
| Inference and checkpoint loading | [generate.py](generate.py) |
| Active DPO checkpoint and weights | [runs/hardened_dpo/checkpoint](runs/hardened_dpo/checkpoint/) |
| Model configuration and tokenizer hash | [runs/hardened_dpo/settings.json](runs/hardened_dpo/settings.json) |
| BPE vocabulary and merges | [runs/hardened_dpo/tokenizer.json](runs/hardened_dpo/tokenizer.json) |
| Pretraining and SFT checkpoint lineage | [runs/hardened_pretrain](runs/hardened_pretrain/), [runs/hardened_sft](runs/hardened_sft/) |
| DPO frozen reference | [runs/hardened_dpo/reference.npz](runs/hardened_dpo/reference.npz) |
| Model facts, scope, and limitations | [MODEL_CARD.md](MODEL_CARD.md) |
| Training data, objectives, and provenance | [TRAINING_BASIS.md](TRAINING_BASIS.md) |
| From-scratch autograd and kernels | [rawllm/tensor.py](rawllm/tensor.py), [rawllm/kernels.py](rawllm/kernels.py) |
| MLA decoder, RMSNorm, RoPE, SwiGLU | [rawllm/model.py](rawllm/model.py) |
| Paged latent KV cache | [rawllm/cache.py](rawllm/cache.py) |
| Byte-level BPE implementation | [rawllm/tokenizer.py](rawllm/tokenizer.py) |
| Complete pretraining/SFT/DPO loops | [train.py](train.py), [rawllm/alignment.py](rawllm/alignment.py) |
| Streaming, packing, and masks | [rawllm/data.py](rawllm/data.py) |
| Optimizer, loss scaling, checkpoint format | [rawllm/optim.py](rawllm/optim.py), [rawllm/safeio.py](rawllm/safeio.py) |
| Raw TCP collectives and distributed examples | [rawllm/distributed.py](rawllm/distributed.py), [distributed_llm3d.py](distributed_llm3d.py) |
| ZeRO sharding and transactional recovery | [rawllm/zero_model.py](rawllm/zero_model.py), [rawllm/sharded_checkpoint.py](rawllm/sharded_checkpoint.py) |
| Equations and manual gradients | [docs/MATHEMATICS.md](docs/MATHEMATICS.md) |
| Full architecture and memory analysis | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Local operation and serving | [docs/OPERATIONS.md](docs/OPERATIONS.md), [serve.py](serve.py) |
| Import and metadata normalization record | [IMPORT_PROVENANCE.json](IMPORT_PROVENANCE.json) |

The detailed framework operations documents assume commands run from `ai/` with the repository virtual environment activated: from the repository root, run `source .venv/bin/activate`, then `cd ai`. Their `python`/`python3` commands will use that environment. Commands below instead run from the repository root with an explicit interpreter.

## Run inference

From the repository root, after installing `requirements-ai.txt` into `.venv`:

```sh
.venv/bin/python ai/generate.py ai/runs/hardened_dpo 'C++: 5 / 2? Answer:' --tokens 24 --temperature 0
```

The dashboard calls the same generation implementation through [backend/model_bridge.py](../backend/model_bridge.py). Greedy decoding is deterministic for a fixed environment and checkpoint. The neural output is labeled experimental and is never used to award XP or silently replace a verified explanation.

The `.npz` payloads are real NumPy archives, not Git LFS pointers or download instructions. Each checkpoint manifest identifies its authoritative payload and SHA-256. The archives also contain optimizer master parameters and Adam moments for resuming; inference selects the model parameter arrays. Obsolete checkpoint generations were not copied.

## Reproduce the three-stage small training experiment

Use a new output path to preserve the shipped checkpoints. These commands run from the repository root:

```sh
.venv/bin/python ai/train.py pretrain --output ai/runs/local-pretrain --steps 3 --max-seconds 90
.venv/bin/python ai/train.py sft --initialize ai/runs/local-pretrain --output ai/runs/local-sft --steps 3 --max-seconds 90
.venv/bin/python ai/train.py dpo --initialize ai/runs/local-sft --output ai/runs/local-dpo --steps 3 --max-seconds 90
.venv/bin/python ai/generate.py ai/runs/local-dpo 'C++' --tokens 24 --temperature 0
```

Each stage uses the original small fixtures in `examples/`, the fixed seed 17, sequence length 64, gradient accumulation of two microbatches, and AdamW. Training loops have bounded local budgets. Time checks occur between optimizer attempts, so an in-flight operation can finish beyond the deadline. DPO is offline preference optimization; no online RL interaction loop is claimed.

## Verify and evaluate

```sh
.venv/bin/python ai/verify.py
.venv/bin/python ai/evaluate.py ai/runs/hardened_dpo --output reports/ai/cpp-quality.json
.venv/bin/python scripts/verify_ai_environment.py --output reports/ai
```

The numerical/process tests, application-in-environment tests, and paired C++ likelihood screen answer different questions. Read [the current reports](../reports/ai/) and [model card](MODEL_CARD.md). Successful tensor arithmetic, checkpoint loading, code execution, or an HTTP response does not establish teaching quality.

`configs/seven_b.json` and `estimate.py --preset 7b` describe a 7,018,450,944-parameter configuration without allocating or training it. Native GPU FlashAttention kernels, RDMA, unified arbitrary 3D-plus-ZeRO scheduling, and production certification are not supplied. The NumPy reference operations and bounded local demonstrations are implemented.

The `evidence/historical` directory preserves prior framework experiment reports. Their run paths and settings hashes refer to the original separate project; they are not current relocated-checkpoint results. Current verification is linked from the repository README. Tensor archives, checkpoint manifests, tokenizers, datasets, and logs were copied without alteration; metadata path normalization is recorded in `IMPORT_PROVENANCE.json`.
