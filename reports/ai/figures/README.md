# Training and test result images

These figures visualize the existing committed evidence. Generating the images does not train the model, change weights, or rerun the test suites.

| Figure | Raster image | Vector image | Evidence |
| --- | --- | --- | --- |
| Training loss | [PNG](training-loss.png) | [SVG](training-loss.svg) | [Pretrain](../../../ai/runs/hardened_pretrain/training.jsonl), [SFT](../../../ai/runs/hardened_sft/training.jsonl), [DPO](../../../ai/runs/hardened_dpo/training.jsonl) |
| Execution and compiler results | [PNG](environment-tests.png) | [SVG](environment-tests.svg) | [Framework](../framework-tests.json), [application](../application-tests.txt), [environment](../environment-validation.json) |
| C++ paired-answer quality | [PNG](cpp-quality.png) | [SVG](cpp-quality.svg) | [Complete quality report](../cpp-quality.json) |

The PNGs are 2240 × 1280 pixels; the SVGs scale without losing resolution. All figures include their source locations, readable labels, and qualifications. The README embeds PNGs for straightforward GitHub rendering. [manifest.json](manifest.json) records SHA-256 hashes for the seven evidence files, six images, and rendering script, plus the Python and Matplotlib versions.

## Interpretation

- Training has three raw loss observations per phase. Lines join those observations without smoothing. Each phase uses a different objective or mask and its own y-axis scale. Decreasing minibatch loss over three updates does not demonstrate convergence, generalization, or successful C++ tutoring. There is no recorded validation-loss curve.
- The 181/181 framework tests, 81/81 application tests, and 36/36 environment checks establish execution behavior within their tested scope. The repeated source-export journey is not added to these counts.
- One neural program was submitted exactly as generated and failed compilation before cases ran. The 5/5 passing program was the separate authored curriculum reference, not a neural generation.
- The paired-answer screen favors the supplied correct answer on 7/18 questions using mean token log probability and 6/18 using summed log probability. This is not generated-answer accuracy. The 95% bootstrap intervals are 16.7–61.1% and 11.1–55.6%, respectively. Both scores must reach 75%, and both interval lower bounds must exceed 50%; the checkpoint fails.

## Regenerate locally

From the repository root, after creating the Python environment described in [the setup guide](../../../environment/README.md):

```sh
.venv/bin/python -m pip install -r requirements-plots.txt
.venv/bin/python scripts/plot_ai_results.py
```

[Matplotlib](../../../requirements-plots.txt) is an optional reporting dependency. It is not imported by model training, inference, the dashboard, or its C++ runner. The renderer uses the noninteractive Agg backend and writes both PNG and SVG; it requires no browser, Docker, GPU, or model service. The [script](../../../scripts/plot_ai_results.py) reads actual log values rather than a manually typed data series. It rejects changed execution/quality outcomes that no longer fit the figure captions, so a later model improvement requires updating those captions rather than silently retaining a stale failure claim.
