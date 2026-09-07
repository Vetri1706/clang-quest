# Training basis and checkpoint provenance

This document audits the exact basis of the three neural checkpoints bundled under `ai/runs/`. It derives counts from the saved manifests, training logs, tokenizer, dataset bytes, and checkpoint arrays. The bundled weights were not retrained during repository integration. Paths below are relative to `ai/`; the [model card](MODEL_CARD.md) describes intended use and limitations.

## Source and import boundary

The source is the NumPy/standard-library framework created earlier in this project. The bundle includes its model, autograd engine, tokenizer, training and alignment code, TCP/distributed demonstrations, environment/serving helpers, tests, configurations, documentation, and original examples. It includes only the latest manifest-selected payload for each of `hardened_pretrain`, `hardened_sft`, and `hardened_dpo`, plus the frozen DPO reference.

[IMPORT_PROVENANCE.json](IMPORT_PROVENANCE.json) records the source digest of each selected imported file. For transformed entries, `sha256` is the original digest and `bundled_sha256` is the digest to verify in this repository. The six transformed files are the three settings and three status files: source/output paths were made relative and obsolete process ids were removed from status. Tensor archives, manifests, tokenizer bytes, datasets, and training logs were not changed.

Historical top-level result JSON files are retained under [evidence/historical](evidence/historical). Some describe earlier runs or separate distributed examples whose checkpoint artifacts are not bundled. They are historical measurements, not a declaration that every referenced run is present. The original whole-project release manifest cannot validate a deliberately selected subset; use the import inventory and fresh repository reports instead.

## Exact datasets

| File | Role | Bytes | Records | SHA-256 |
| --- | --- | ---: | --- | --- |
| [examples/corpus.txt](examples/corpus.txt) | Tokenizer training and pretraining | 971 | 10 prose lines; one streamed document at the default chunk size | `588eb2b4d1c2be975db6124c705e66ac18217f9814fd84486e4786983a793130` |
| [examples/sft.jsonl](examples/sft.jsonl) | SFT | 264 | 3 prompt/response records | `956832bd33fa796d32e0fd8d336f8c1bc9bb29aa5159f0674ebb5364de2afc95` |
| [examples/preferences.jsonl](examples/preferences.jsonl) | Offline DPO | 222 | 3 chosen/rejected pairs | `44ce38b27adcd33bad0c8cd5bcdbdbe45b4b0e2a9dcdd840079305fedb028286` |
| [examples/cpp_eval.jsonl](examples/cpp_eval.jsonl) | Evaluation only | 4,162 | 18 paired questions | `505d503f1666a4fae0bb717f17b098d85f55580f743b3ddc6eca3fac52d70b82` |

The training files total 1,457 raw bytes including JSON syntax. They are small project-authored examples; no third-party dataset download or imported pretrained model is recorded. The prose covers elementary C++ arithmetic, RAII/lifetime, compilers and LLVM IR, game loops, gradients, evaluation, networking, and checkpoints. The SFT records concern integer division, RAII, and LLVM IR. DPO pairs prefer `2` over `3` for `5/2`, object-lifetime cleanup over a garbage-collector description for RAII, and compiler infrastructure over a game-texture description for LLVM.

The evaluation file is excluded from all three recorded training dataset digests and the tokenizer source. It is a small original probe with conceptual overlap with the training material, not a deduplicated or representative benchmark. It contains two questions per combination of beginner/intermediate/advanced and C++/compilers/game development. No free-form generated code is executed during its likelihood screen.

Questline's separate 24 mission definitions, 24 reference cards, learner code, saved chat messages, and compiler test results are not inputs to this neural training chain. Their presence in the same repository does not imply that these weights learned them.

## Tokenizer and packing

The tokenizer was trained deterministically on all 971 bytes of `corpus.txt`, with a 65,536-byte input cap, requested vocabulary 288, and minimum pair frequency two. The initial 259 symbols are PAD=0, BOS=1, EOS=2, and all 256 byte values. Training added 29 merges, choosing the most frequent adjacent pair and breaking frequency ties by pair ids. Encoding repeatedly applies the lowest-ranked available merge. UTF-8 bytes provide a complete fallback; byte completeness is not multilingual competence.

All three runs have byte-identical `tokenizer.json` files with SHA-256:

```text
58b7f45d9ccdd4c22ec61d27b93504147eeb655987e9046f0b5f2d7309cba233
```

The corpus encodes to 684 non-special tokens. With BOS/EOS and 64-position packing, one full pass yields 11 windows and 685 possible supervised next-token targets. A one-token context overlap links adjacent windows without repeating targets. Cross-document transitions and padding are masked from the loss; attention is causal and blocked across document boundaries. The default text reader treats emitted chunks as documents, not individual lines.

For SFT, prompt and response are encoded separately so a BPE merge cannot cross their boundary. BOS belongs to the unsupervised prompt; response tokens and EOS receive supervision. Three records pack into three windows per pass, with 33, 33, and 17 supervised targets. DPO forms each chosen/rejected completion separately and rejects overlong completions instead of truncating their responses.

## Recorded optimization recipe

Every stage used seed 17, FP32 training, a 64-position window, accumulation of two training units per update, and fixed learning rate `0.0003`. AdamW used betas `(0.9, 0.999)`, epsilon `1e-8`, and decoupled weight decay `0.01`. The accumulated global gradient norm was clipped to 1.0 before each update. No learning-rate schedule or dropout is implemented in this recipe.

For pretraining and SFT, each microbatch mean loss is multiplied by its supervised-token count divided by the total supervised-token count across the accumulation group. This produces a token-weighted update even when windows contain different response-mask masses. The loss is next-token negative log-likelihood in natural logarithms; pretraining derives its targets from the text itself, while SFT uses explicit prompt/response supervision.

AdamW maintains a model tensor, an FP32 master tensor, and FP32 first/second moments for each unique parameter. For a gradient \(g_t\), the implementation computes

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
\]

All candidate state arrays are checked before an optimizer update is committed. The three final checkpoints record FP32 loss scale 1.0, three good scaling steps, and no overflow skips. FP16/BF16 support elsewhere in the framework is rounding emulation with FP32 compute/master state; these published weights were not trained using either reduced-precision option.

### Actual exposure, not configured capacity

| Run | Final step / updates / skips | Consumed units | Exact consumed supervision |
| --- | --- | ---: | --- |
| `hardened_pretrain` | 3 / 3 / 0 | 6 packed windows | 6 × 64 = 384 targets |
| `hardened_sft` | 3 / 3 / 0 | 6 packed windows | 33 + 33 + 17 + 33 + 33 + 17 = 166 response/EOS targets |
| `hardened_dpo` | 3 / 3 / 0 | 6 pairs | The 3 preference records repeated twice |

The pretraining run stops before its first full 11-window pass. SFT and DPO cycle twice through their finite streams. In DPO, chosen response/EOS lengths are 2, 18, and 22 tokens; rejected lengths are 2, 16, and 13. Across two passes, the policy scores 84 chosen and 62 rejected response tokens. The frozen reference scores the same responses, but those evaluations do not add training examples or update reference weights.

`training.jsonl` has three records per stage. Each records the step, consumed stream units, updates, skipped updates, loss, pre-clipping gradient norm, scale, finite flag, and any forward error. A successful numerical update demonstrates execution of the objective, not mastery of the examples.

## DPO and the meaning of RL here

The policy begins DPO at the final SFT weights. A separate frozen reference is saved at that exact point; an array-by-array comparison confirms it equals the bundled final SFT parameters. Neither reference parameters nor reference log probabilities receive policy gradients.

For prompt \(x\), chosen completion \(y^+\), rejected completion \(y^-\), and fixed reference \(\pi_{\rm ref}\), the implemented objective is

\[
-\log\sigma\left(0.1\left[
\log\pi_\theta(y^+\mid x)-\log\pi_\theta(y^-\mid x)
-\log\pi_{\rm ref}(y^+\mid x)+\log\pi_{\rm ref}(y^-\mid x)
\right]\right).
\]

Response scores are sums of masked token log probabilities, including EOS; they are not normalized by answer length. Stable softplus implements negative log-sigmoid. The two pairs in an accumulation group receive equal weight.

This is offline preference optimization. The recorded runs have no online reward signal, learned reward model, policy/environment rollout loop, PPO update, or C++-execution reward. The repository's compiler environment and distributed numerical demonstrations are available implementation components, not evidence of environment-based RL for these checkpoints.

## Checkpoint state, lineage, and exact identities

A checkpoint contains 30 named model tensors and three matching optimizer groups, for 120 arrays total. Each of the four groups contains 118,624 data bytes; NPZ headers and ZIP metadata bring each saved file to 504,018 bytes. Visible parameter values equal FP32 master values in these final FP32 snapshots. `gradients` is empty because the trainer clears gradients before saving complete optimizer boundaries.

| Run | Checkpoint payload SHA-256 | Manifest SHA-256 |
| --- | --- | --- |
| Pretraining | `5a8600333b3116953ed6dd8e79c1545903ca3a722999e1ed5269d47df7d0034a` | `82bbb961e763c1b5824a9874225c48810a55ebc6de84d0b755f557f50d9fa632` |
| SFT | `df087ebe3fdce9ad99bcb6e1f6899642ad5fcbbf28c66e8ef60cc1430ba4b7e5` | `e61b3777e66dabcc8a18b45fee4e632685135ff755e603340567231fb2741bb7` |
| DPO | `f803d1cf276ae9bb02858d37f516a2c3aa7a455d9213d03ffe62318bd4e51bf5` | `a2e98920c25c00dcf45efe4196cd3568c2a0275538bb3e096f666535b9371605` |

The frozen DPO `reference.npz` is 126,436 bytes, contains 30 model arrays, and has SHA-256 `1e30f681518e38117145c247f2607001e93d1a62eedd37aa293c1ff8838beb26`.

Each stage uses a fresh AdamW optimizer. `--initialize` copies model and tokenizer but resets optimizer, loss scaler, RNG, and counters for the new phase. `--resume` instead restores optimizer moments and clocks, RNG, loss scaler, counters, and deterministic stream position. Saved counters describe complete attempts: `samples_consumed = step × accumulation`, and `step = updates + skipped_updates`.

The settings record the stage-to-stage initialization paths. They do not contain a cryptographically signed ancestry chain; the logs, saved settings, unchanged selected artifacts, and frozen-reference equality establish the available local provenance. File hashes identify these artifacts, not their quality or independently authenticated authorship.

## Portability and validation dependencies

- Keep each `checkpoint/manifest.json` with the exact basename it names. The loader's archive inventory expects the optimizer groups even when inference selects only model arrays; extracting parameters into a new archive requires a different export format/loader.
- Keep `settings.json` and byte-identical `tokenizer.json`. The tokenizer digest is checked before load. Copy DPO `reference.npz` for resume; it is not needed to decode the final policy.
- Training resume checks phase, dataset digest, precision, sequence length, accumulation, and DPO beta. An identical dataset may be relocated: the digest is the identity check. The normalized metadata paths are descriptive; training defaults resolve shipped examples relative to `train.py`.
- Metadata path normalization changes settings/status bytes. The original final-DPO settings digest was `1a544ab2c787d386d9c4c4abe7558561b9875f1ef17a1859fe492acda7c58d32`; the bundled settings digest is `0170f5bb6fc16b0cfe75154f6e1d680daebddbff4c5a224c2e5b2275fd5c8ba7`. The [fresh quality report](../reports/ai/cpp-quality.json) records the latter, while historical reports retain the former. The weight and manifest hashes remain unchanged.
- Payload names and archive metadata are generated during checkpoint creation, so a new equivalent training run need not produce byte-identical NPZ filenames or file hashes. Compare parameter arrays and recorded metrics with appropriate numerical tolerances across environments.
- The NumPy framework is distinct from the macOS C++ sandbox. Python 3.14.6/NumPy 2.4.6 on Darwin arm64 is the original measured environment; `requirements-tested.txt` pins NumPy 2.4.6, which requires Python 3.11 or newer; the dashboard alone has a separate Python 3.10 minimum. No trained 7B weights or native GPU kernels are bundled.

For a new three-stage experiment, run from `ai/` and use new output directories:

```sh
../.venv/bin/python train.py pretrain --output runs/local-pretrain --steps 3 --accumulation 2 --sequence-length 64 --precision fp32 --seed 17
../.venv/bin/python train.py sft --initialize runs/local-pretrain --output runs/local-sft --steps 3 --accumulation 2 --sequence-length 64 --precision fp32 --seed 17
../.venv/bin/python train.py dpo --initialize runs/local-sft --output runs/local-dpo --steps 3 --accumulation 2 --sequence-length 64 --precision fp32 --seed 17 --beta 0.1
```

These commands describe the saved recipe; they do not change or improve the bundled evidence. Continuing an experiment requires a copied run directory and `--resume --steps N`, where N is the total target attempt count, not an increment. The frozen reference must remain fixed across DPO resume.

The [fresh C++ evaluation](../reports/ai/cpp-quality.json) failed with mean preference accuracy 38.89%, summed preference accuracy 33.33%, and preferred-response perplexity 278.8523. All 18 examples were scored. The [fresh framework verification](../reports/ai/framework-tests.json) reports 181 passing implementation tests, zero failures/errors/skips, and the same audited Python-source digest as the original framework; its scope is numerical and local process behavior. Neither successful inference nor passing systems tests establishes tutor competence.
