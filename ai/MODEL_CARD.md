# Model card: Questline experimental NumPy decoder

The bundled neural model is a **29,656-parameter research model** trained through three short stages: three pretraining updates, three supervised fine-tuning updates, and three direct preference optimization updates. Its saved weights support real local inference. **It failed the C++ likelihood quality gate and is not a competent general-purpose C++ tutor.** Questline's default study mentor remains the separate curated guide and compiler-feedback system.

Paths and links in this document are relative to the repository's `ai/` directory. The exact training inputs, checkpoint identities, and optimization semantics are documented in [TRAINING_BASIS.md](TRAINING_BASIS.md).

## Identity and architecture

| Property | Bundled model |
| --- | --- |
| Final policy | `runs/hardened_dpo` |
| Earlier stages | `runs/hardened_pretrain`, `runs/hardened_sft` |
| Unique trainable parameters | **29,656**, independently counted from the checkpoint arrays |
| Parameter tensors | 30; tied input/output embedding appears once |
| Storage and recorded training precision | FP32 |
| Parameter data bytes | 118,624, excluding optimizer, runtime, cache, and activations |
| Transformer blocks | 2 |
| Residual width / attention heads | 32 / 4 |
| Query latent rank / KV latent rank | 16 / 12 |
| Per-head content / rotary / value widths | 8 / 8 / 8 |
| SwiGLU intermediate width | 64 |
| Vocabulary | 288: three special symbols, all 256 bytes, and 29 BPE merges |
| Maximum configured context | 256 tokens |
| Recorded training window | 64 input/target positions per microbatch |
| Position and normalization | RoPE base 10,000; RMSNorm epsilon `1e-6` |
| Initialization | Seed 17; matrix entries sampled from Normal(0, 0.02), normalization scales initialized to one |

The parameter count is `288 × 32 + 2 × 10,204 + 32 = 29,656`. The implementation is a decoder with multi-head latent attention, RMSNorm, rotary positions, residual connections, and SwiGLU feed-forward blocks. It uses a custom reverse-mode autodifferentiation engine, NumPy arithmetic, tiled exact attention, and a compressed paged latent cache for incremental decoding. The language-model output projection shares the embedding weights.

The saved [run settings](runs/hardened_dpo/settings.json) are authoritative. The illustrative `configs/tiny.json` has a different vocabulary and context limit; it is not an exact description of these weights. The 256-token context is an implementation limit, not evidence of long-context competence after training on 64-position windows.

## What was trained

The files are a sequential lineage recorded as random initialization → `hardened_pretrain` → `hardened_sft` → `hardened_dpo`. Initializing a new phase copies model weights and tokenizer, then starts a fresh optimizer and counters. The final checkpoint therefore says step 3, not step 9; nine successful updates occurred across the recorded chain.

| Stage | Successful updates | Consumed training units | Supervision consumed | First → final logged loss |
| --- | ---: | ---: | --- | --- |
| Pretraining | 3 | 6 packed windows | 384 next-token targets | 5.6697669029 → 5.6444213390 |
| SFT | 3 | 6 packed windows | 166 response/EOS targets | 5.6612817013 → 5.6130114746 |
| DPO | 3 | 6 preference pairs | All 3 pairs visited twice | 0.6931471806 → 0.6634032500 |

All three stages report zero skipped updates. The pretraining corpus is 971 bytes and yields 11 packed windows per pass. **Only its first six windows were consumed by the recorded pretraining run.** Tokenizer construction did read the entire 971-byte text. SFT and DPO each completed two passes over their tiny respective inputs. These loss values measure different objectives on different microbatches and are not directly comparable across stages.

Training data comprises ten short project-authored prose lines, three prompt/response SFT records, and three preference pairs. It is not a crawl, an imported public model, or an ingestion of all free datasets. The larger Questline challenge catalog, reference cards, and user conversations were not training inputs for this neural checkpoint.

DPO is offline preference optimization against a frozen reference. There was no online environment rollout, PPO training, learned reward model, or interaction-based reinforcement-learning episode in this recorded chain. The C++ runner and distributed training demonstrations are separate components; they did not provide rewards to these weights.

## Measured quality

The [fresh evaluation of the relocated checkpoint](../reports/ai/cpp-quality.json) reproduces the final model's earlier results:

| Metric | Result |
| --- | --- |
| Gate | **Fail** |
| Paired questions scored | 18 of 18; zero omitted or rejected |
| Coverage | 3 levels × 3 domains × 2 questions |
| Mean response-token preference accuracy | 7/18 = 38.89% |
| Summed response-log-probability preference accuracy | 6/18 = 33.33% |
| Preferred-response mean NLL | 5.6306823570 nats/token |
| Preferred-response token perplexity | 278.8523295705 |
| Preferred-response tokens scored | 423, including EOS |
| 95% bootstrap interval, mean preference | 16.67%–61.11% |
| 95% bootstrap interval, summed preference | 11.11%–55.56% |

The local gate requires at least 18 questions, at least two per level/domain cell, preference accuracy of at least 75% for both scoring conventions, and bootstrap lower bounds above 50%. Both accuracy criteria and both lower-bound criteria failed. There were no preference ties. Bootstrap estimates use 1,000 deterministic resamples with seed 173 and describe sensitivity to this small prompt set.

This is a **teacher-forced paired-likelihood screen**: supplied correct and incorrect answers are scored token by token. It does not test independent free-form explanation, generated-code compilation, student learning outcomes, comprehensive safety, or general programming competence. Correct preceding answer tokens are supplied during scoring. Summed and mean scores also treat answer length differently. The [fresh framework check](../reports/ai/framework-tests.json) reports 181 passing tests with zero failures, errors, or skips. These establish functioning numerical and systems code, not model expertise.

The separate historical `cpp_evaluation.json` evaluated the older two-step `runs/pretrain` checkpoint, which is not one of the three bundled hardened runs. It must not be presented as a controlled before/after comparison of this three-checkpoint chain.

## Intended use and interface

Use the model to inspect an end-to-end NumPy training/inference implementation, reproduce checkpoint loading, investigate token probabilities, or run small controlled experiments. Treat generated text as experimental output. Use the grounded Questline guide and actual compiler/test feedback for learning assistance.

From `ai/`, with the tested NumPy dependency installed:

```sh
../.venv/bin/python generate.py runs/hardened_dpo "Explain C++ integer division:" --tokens 16 --temperature 0
```

The generator returns a JSON object explicitly labeled `unvalidated_tiny_model_output`. Greedy decoding uses temperature zero; positive temperature samples from the next-token distribution with the generator's seeded RNG. PAD and BOS are suppressed as generated tokens, EOS terminates generation, and prompt plus output must fit the configured context. This is a short text continuation interface, not a general chat protocol or tool-using agent.

The Python implementation requires NumPy and the standard library; no PyTorch, TensorFlow, JAX, Hugging Face, or LangChain model code is used. The original measured environment was Python 3.14.6, NumPy 2.4.6, macOS/Darwin arm64, with BLAS thread counts constrained to one for bounded local runs. [requirements-tested.txt](requirements-tested.txt) pins NumPy 2.4.6, whose package metadata requires Python 3.11 or newer. This pinned AI environment has a higher Python minimum than the standalone grounded dashboard. Other NumPy/BLAS/platform combinations may produce small numerical differences. The separate restricted C++ execution environment remains macOS-specific.

## Artifacts and integrity

Each checkpoint archive has 120 arrays: 30 model tensors plus the corresponding FP32 master weights, first moments, and second moments. Its total file size is 504,018 bytes; model tensor data alone is 118,624 bytes. Optimizer-state arrays are training state, not additional model parameters. Gradients were cleared before final checkpointing.

| Stage | Manifest-selected archive | SHA-256 |
| --- | --- | --- |
| Pretraining | `weights-6c75b0af8b4f49749e8245c276ad3cf5.npz` | `5a8600333b3116953ed6dd8e79c1545903ca3a722999e1ed5269d47df7d0034a` |
| SFT | `weights-56b3858c0f4b4752bc3ec1c097e31c94.npz` | `df087ebe3fdce9ad99bcb6e1f6899642ad5fcbbf28c66e8ef60cc1430ba4b7e5` |
| DPO | `weights-09437d05fd214570a2786922d365d1d1.npz` | `f803d1cf276ae9bb02858d37f516a2c3aa7a455d9213d03ffe62318bd4e51bf5` |

Each archive lives under its run's `checkpoint/` directory. Loaders verify the named payload, its digest, tensor schema, dtypes, finite values, and tokenizer digest. NumPy archives are loaded without pickle. Checksums establish file identity and detect corruption; they are not independent authentication against replacement of both metadata and weights.

The DPO [reference archive](runs/hardened_dpo/reference.npz) contains a frozen copy of the final SFT model, verified array-for-array. It is required for exact DPO resume, although final-policy inference does not use it. [IMPORT_PROVENANCE.json](IMPORT_PROVENANCE.json) distinguishes unchanged imported tensors/logs from normalized metadata. Historical evidence describes the original files; the fresh evaluation records the relocated settings hash.

## What the 7B configuration means

[configs/seven_b.json](configs/seven_b.json) describes an architecture with **7,018,450,944 parameters**, computed from tensor shapes without allocation. No weights for that configuration were allocated, trained, or bundled. The local `Transformer` constructor's default parameter allocation ceiling rejects a model that large.

FP32 parameters alone would require 28,073,803,776 bytes, before gradients, optimizer state, activations, or caches. The framework's capacity reports and distributed examples are small-scale implementation and arithmetic demonstrations. They do not establish a trained 7B model, native GPU performance, production-scale throughput, or unrestricted knowledge.
