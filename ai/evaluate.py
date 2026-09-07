"""Evaluate a saved local run on original held-out C++ likelihood pairs."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate import load_run
from rawllm.evaluation import EvaluationLimits, evaluate, load_examples
from rawllm.runtime import atomic_json
from rawllm.safeio import read_json, regular_file


def digest(path):
    result = hashlib.sha256()
    with regular_file(path, 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--examples", type=Path, default=Path(__file__).parent / "examples/cpp_eval.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-sequence-tokens", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=173)
    parser.add_argument("--require-pass", action="store_true", help="Exit 2 if the limited likelihood screen does not pass")
    args = parser.parse_args()
    try:
        limits = EvaluationLimits(max_examples=args.max_examples, max_sequence_tokens=args.max_sequence_tokens,
                                  max_seconds=args.max_seconds, bootstrap_samples=args.bootstrap_samples,
                                  bootstrap_seed=args.bootstrap_seed)
        artifacts = {"checkpoint_manifest_sha256": args.run / "checkpoint/manifest.json",
                     "tokenizer_sha256": args.run / "tokenizer.json",
                     "run_settings_sha256": args.run / "settings.json",
                     "evaluation_sha256": args.examples}
        before = {name: digest(path) for name, path in artifacts.items()}
        manifest = read_json(args.run / "checkpoint/manifest.json")
        examples = load_examples(args.examples)
        model, tokenizer = load_run(args.run)
        report = evaluate(model, tokenizer, examples, limits)
        if before != {name: digest(path) for name, path in artifacts.items()}:
            raise ValueError("An evaluation artifact changed during the run")
        report["reproducibility"] = {"checkpoint_payload_sha256": manifest["sha256"],
                                     **before,
                                     "parameter_count": model.config.parameter_count,
                                     "numpy_version": np.__version__, "precision": model.precision,
                                     "checkpoint_counters": manifest.get("counters", {})}
        report["split"] = {"role": "evaluation_only", "training_exclusion": "The shipped evaluation JSONL is not a default training input; a supplied/custom run's training history must be audited separately.",
                           "question_authorship": "Original short paired questions authored for this local framework.",
                           "fact_check_references": ["https://llvm.org/docs/LangRef.html#phi-instruction",
                                                     "https://eel.is/c++draft/vector.modifiers"]}
        atomic_json(args.output, report)
    except (OSError, ValueError, TypeError, KeyError, MemoryError):
        parser.exit(2, "Evaluation failed: verify the trusted run, exact evaluation schema, and resource limits.\n")
    print(json.dumps({"status": "evaluated", "scored_examples": report["scored_examples"],
                      "rejected_examples": len(report["rejected_examples"]),
                      "chosen_token_perplexity": report["overall"]["chosen_token_perplexity"],
                      "mean_preference_accuracy": report["overall"]["mean_preference_accuracy"],
                      "sum_preference_accuracy": report["overall"]["sum_preference_accuracy"],
                      "likelihood_gate": report["gate"]["status"],
                      "freeform_tutor_competence_assessed": False}, allow_nan=False))
    if args.require_pass and report["gate"]["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
