"""Bounded, response-masked likelihood evaluation for an original C++ prompt set.

This is a deterministic paired-likelihood screen. It does not execute generated
code, measure explanation quality, or establish freeform tutoring competence.
The bootstrap describes resampling sensitivity within the supplied prompt set;
its intervals do not establish representativeness of all programming questions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .kernels import log_softmax
from .safeio import regular_file
from .tensor import no_grad


LEVELS = ("beginner", "intermediate", "advanced")
DOMAINS = ("cpp", "compilers", "game_development")
FIELDS = {"id", "level", "domain", "prompt", "chosen", "rejected"}


@dataclass(frozen=True)
class EvaluationLimits:
    max_examples: int = 64
    max_sequence_tokens: int = 256
    max_logits_bytes: int = 32 * 1024**2
    max_seconds: float = 30.0
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 173
    bootstrap_min_examples: int = 8

    def __post_init__(self):
        if type(self.max_examples) is not int or not 1 <= self.max_examples <= 1024:
            raise ValueError("Evaluate between 1 and 1024 examples")
        if type(self.max_sequence_tokens) is not int or not 1 <= self.max_sequence_tokens <= 4096:
            raise ValueError("Evaluation sequences must be bounded to 1..4096 tokens")
        if type(self.max_logits_bytes) is not int or not 1 <= self.max_logits_bytes <= 512 * 1024**2:
            raise ValueError("Logit arrays must be bounded to at most 512 MiB")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 300:
            raise ValueError("Evaluation time must be bounded to 0..300 seconds")
        if type(self.bootstrap_samples) is not int or not 100 <= self.bootstrap_samples <= 10000:
            raise ValueError("Use 100..10000 bootstrap replicates")
        if type(self.bootstrap_min_examples) is not int or not 2 <= self.bootstrap_min_examples <= 1024:
            raise ValueError("Bootstrap needs at least two examples")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**63:
            raise ValueError("Bootstrap seed must be a nonnegative 63-bit integer")


@dataclass(frozen=True)
class GateCriteria:
    minimum_examples: int = 18
    minimum_per_level_domain: int = 2
    minimum_preference_accuracy: float = 0.75
    minimum_bootstrap_lower_bound: float = 0.5

    def __post_init__(self):
        if type(self.minimum_examples) is not int or self.minimum_examples < 2:
            raise ValueError("A likelihood gate requires at least two examples")
        if type(self.minimum_per_level_domain) is not int or self.minimum_per_level_domain < 1:
            raise ValueError("Every level/domain cell needs at least one example")
        for value in (self.minimum_preference_accuracy, self.minimum_bootstrap_lower_bound):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Gate preference thresholds must be in [0,1]")


def validate_example(record):
    if not isinstance(record, dict) or set(record) != FIELDS or any(not isinstance(v, str) for v in record.values()):
        raise ValueError("Evaluation rows require exactly id, level, domain, prompt, chosen, and rejected strings")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", record["id"]):
        raise ValueError("Invalid evaluation example ID")
    if record["level"] not in LEVELS or record["domain"] not in DOMAINS:
        raise ValueError("Unknown evaluation level or domain")
    for key in ("prompt", "chosen", "rejected"):
        if not record[key].strip() or len(record[key].encode("utf-8")) > 2048:
            raise ValueError("Evaluation text fields must be nonempty and at most 2048 UTF-8 bytes")
    if record["chosen"] == record["rejected"]:
        raise ValueError("Chosen and rejected responses must differ")
    return dict(record)


def load_examples(path, max_bytes=1024 * 1024, max_records=1024):
    """Read a bounded original-evaluation JSONL artifact with exact field names."""
    if type(max_records) is not int or not 1 <= max_records <= 1024:
        raise ValueError("Evaluation inventory is limited to 1024 records")
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate evaluation JSON field")
            result[key] = value
        return result
    records, identifiers, prompts = [], set(), set()
    with regular_file(path, max_bytes) as stream:
        consumed = 0
        while True:
            line = stream.readline(8193)
            if not line:
                break
            consumed += len(line)
            if len(line) > 8192 or consumed > max_bytes:
                raise ValueError("Evaluation artifact or row exceeds its byte limit")
            if not line.strip():
                continue
            if len(records) == max_records:
                raise ValueError("Evaluation artifact exceeds its record limit")
            try:
                record = validate_example(json.loads(line, object_pairs_hook=object_pairs))
            except (UnicodeError, RecursionError) as error:
                raise ValueError("Invalid evaluation JSON encoding") from error
            if record["id"] in identifiers or record["prompt"] in prompts:
                raise ValueError("Evaluation IDs and prompts must be unique")
            identifiers.add(record["id"])
            prompts.add(record["prompt"])
            records.append(record)
    if not records:
        raise ValueError("Evaluation artifact is empty")
    return records


def response_batch(tokenizer, prompt, response, max_sequence_tokens):
    """Encode prompt and answer separately; supervise response tokens and EOS.

    Input token i predicts combined token i+1. Therefore the first supervised
    position is len(prompt_with_BOS)-1, not len(prompt_with_BOS). Overlong pairs
    are rejected in full; neither prompt nor response is truncated.
    """
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(response, add_eos=True)
    combined = prompt_ids + answer_ids
    if len(combined) - 1 > max_sequence_tokens:
        raise ValueError("tokenizer_context_limit")
    tokens = np.asarray(combined, dtype=np.int64)
    inputs, targets = tokens[:-1][None, :], tokens[1:][None, :]
    mask = (np.arange(len(combined) - 1) >= len(prompt_ids) - 1)[None, :]
    return inputs, targets, mask


def token_likelihood(logits, targets, response_mask):
    """Return response-only summed/mean log probabilities and token perplexity."""
    logits, targets, mask = np.asarray(logits), np.asarray(targets), np.asarray(response_mask)
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("Likelihood expects logits [B,T,V] and matching targets/mask")
    if targets.dtype.kind not in "iu" or np.any(targets < 0) or np.any(targets >= logits.shape[-1]):
        raise ValueError("Likelihood targets must be valid integer token IDs")
    if not np.isfinite(logits).all() or not np.isfinite(mask).all() or np.any((mask != 0) & (mask != 1)):
        raise ValueError("Logits must be finite and the response mask binary")
    count = int(mask.sum())
    if count == 0:
        raise ValueError("Likelihood needs at least one response target")
    selected = np.take_along_axis(log_softmax(logits.astype(np.float64)), targets[..., None], axis=-1)[..., 0]
    total = float(selected[mask.astype(bool)].sum(dtype=np.float64))
    mean = total / count
    perplexity = math.exp(-mean) if -mean < math.log(np.finfo(np.float64).max) else None
    return {"response_tokens": count, "sum_log_probability": total,
            "mean_log_probability": mean, "mean_token_nll": -mean,
            "token_perplexity": perplexity}


def _bootstrap(values, limits, name, weights=None, deadline=None):
    values = np.asarray(values, dtype=np.float64)
    if values.size < limits.bootstrap_min_examples:
        return None
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    # Stable named streams prevent adding another breakdown from changing an
    # existing interval. Each replicate holds only O(example_count) integers.
    tag = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([limits.bootstrap_seed, tag]))
    estimates = np.empty(limits.bootstrap_samples, dtype=np.float64)
    for index in range(limits.bootstrap_samples):
        if deadline is not None and index % 64 == 0 and time.monotonic() >= deadline:
            return None
        selected = rng.integers(0, values.size, size=values.size)
        estimates[index] = values[selected].mean() if weights is None else np.sum(values[selected] * weights[selected]) / np.sum(weights[selected])
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper), "confidence_level": 0.95,
            "method": "deterministic_nonparametric_percentile_bootstrap",
            "replicates": limits.bootstrap_samples, "examples": int(values.size)}


def summarize(rows, limits, name="overall", deadline=None):
    if not rows:
        return {"examples": 0, "chosen_response_tokens": 0, "chosen_mean_token_nll": None,
                "chosen_token_perplexity": None, "mean_preference_accuracy": None,
                "sum_preference_accuracy": None, "mean_preference_ci95": None,
                "sum_preference_ci95": None, "mean_nll_ci95": None,
                "mean_preference_ties": 0, "sum_preference_ties": 0}
    tokens = sum(row["chosen"]["response_tokens"] for row in rows)
    total_nll = -sum(row["chosen"]["sum_log_probability"] for row in rows)
    mean_nll = total_nll / tokens
    mean_votes = [row["mean_preference_credit"] for row in rows]
    sum_votes = [row["sum_preference_credit"] for row in rows]
    return {"examples": len(rows), "chosen_response_tokens": tokens,
            "chosen_mean_token_nll": mean_nll,
            "chosen_token_perplexity": math.exp(mean_nll) if mean_nll < 709 else None,
            "mean_preference_accuracy": float(np.mean(mean_votes)),
            "sum_preference_accuracy": float(np.mean(sum_votes)),
            "mean_preference_ci95": _bootstrap(mean_votes, limits, name + ":mean_preference", deadline=deadline),
            "sum_preference_ci95": _bootstrap(sum_votes, limits, name + ":sum_preference", deadline=deadline),
            "mean_nll_ci95": _bootstrap([row["chosen"]["mean_token_nll"] for row in rows], limits, name + ":token_weighted_nll",
                                        weights=[row["chosen"]["response_tokens"] for row in rows], deadline=deadline),
            "mean_preference_ties": sum(v == 0.5 for v in mean_votes),
            "sum_preference_ties": sum(v == 0.5 for v in sum_votes)}


def likelihood_gate(summary, coverage, rejected_count, criteria):
    reasons = []
    if summary["examples"] < criteria.minimum_examples:
        reasons.append("too_few_scored_examples")
    if any(coverage[level][domain] < criteria.minimum_per_level_domain for level in LEVELS for domain in DOMAINS):
        reasons.append("insufficient_level_domain_coverage")
    if rejected_count:
        reasons.append("evaluation_examples_were_not_scored")
    if reasons:
        status = "insufficient_coverage"
    else:
        for method in ("mean", "sum"):
            if summary[method + "_preference_accuracy"] < criteria.minimum_preference_accuracy:
                reasons.append(method + "_preference_accuracy_below_threshold")
            interval = summary[method + "_preference_ci95"]
            if interval is None:
                reasons.append(method + "_preference_interval_unavailable")
            elif interval["lower"] <= criteria.minimum_bootstrap_lower_bound:
                reasons.append(method + "_preference_interval_not_above_threshold")
        status = "fail" if reasons else "pass"
    return {"status": status, "scope": "paired_teacher_forced_likelihood_screen_only",
            "freeform_tutor_competence_assessed": False, "criteria": asdict(criteria), "reasons": reasons}


def evaluate(model, tokenizer, examples, limits=None, criteria=None):
    limits, criteria = limits or EvaluationLimits(), criteria or GateCriteria()
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Evaluation tokenizer/model vocabularies must agree")
    if model.precision != "fp32":
        raise ValueError("This evaluation recipe fixes computation to fp32")
    if not isinstance(examples, (list, tuple)) or not 1 <= len(examples) <= 1024:
        raise ValueError("Evaluation requires a bounded nonempty list of at most 1024 examples")
    examples = [validate_example(record) for record in examples]
    if len({record["id"] for record in examples}) != len(examples) or len({record["prompt"] for record in examples}) != len(examples):
        raise ValueError("Evaluation IDs and prompts must be unique")
    started = time.monotonic()
    deadline = started + limits.max_seconds
    maximum = min(model.config.max_seq_len, limits.max_sequence_tokens)
    logit_itemsize = max((parameter.dtype.itemsize for parameter in getattr(model, "params", {}).values()), default=4)
    rows, rejected = [], []
    for index, record in enumerate(examples):
        reason = None
        if index >= limits.max_examples:
            reason = "example_limit"
        elif time.monotonic() >= deadline:
            reason = "time_limit"
        if reason is not None:
            rejected.append({"id": record["id"], "reason": reason})
            continue
        batches = {}
        try:
            for label in ("chosen", "rejected"):
                batches[label] = response_batch(tokenizer, record["prompt"], record[label], maximum)
        except ValueError as error:
            if str(error) != "tokenizer_context_limit":
                raise
            rejected.append({"id": record["id"], "reason": "tokenizer_context_limit"})
            continue
        scores = {}
        for label, (inputs, targets, mask) in batches.items():
            if inputs.size * model.config.vocab_size * logit_itemsize > limits.max_logits_bytes:
                reason = "logits_memory_limit"
                break
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            with no_grad():
                logits = model(inputs).data
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            scores[label] = token_likelihood(logits, targets, mask)
        if reason is not None:
            rejected.append({"id": record["id"], "reason": reason})
            continue
        mean_margin = scores["chosen"]["mean_log_probability"] - scores["rejected"]["mean_log_probability"]
        sum_margin = scores["chosen"]["sum_log_probability"] - scores["rejected"]["sum_log_probability"]
        def credit(margin):
            return 1.0 if margin > 1e-12 else 0.0 if margin < -1e-12 else 0.5
        rows.append({"id": record["id"], "level": record["level"], "domain": record["domain"],
                     **scores, "mean_log_probability_margin": mean_margin,
                     "sum_log_probability_margin": sum_margin,
                     "mean_preference_credit": credit(mean_margin), "sum_preference_credit": credit(sum_margin)})
    summary = summarize(rows, limits, deadline=deadline)
    coverage = {level: {domain: sum(row["level"] == level and row["domain"] == domain for row in rows)
                        for domain in DOMAINS} for level in LEVELS}
    rejection_counts = {reason: sum(row["reason"] == reason for row in rejected)
                        for reason in ("tokenizer_context_limit", "example_limit", "time_limit", "logits_memory_limit")}
    by_level = {level: summarize([row for row in rows if row["level"] == level], limits, "level:" + level, deadline)
                for level in LEVELS}
    by_domain = {domain: summarize([row for row in rows if row["domain"] == domain], limits, "domain:" + domain, deadline)
                 for domain in DOMAINS}
    return {"format": "rawllm-cpp-likelihood-evaluation", "version": 1,
            "available_examples": len(examples), "scored_examples": len(rows), "rejected_examples": rejected,
            "rejection_counts": rejection_counts, "effective_context_tokens": maximum,
            "estimated_logit_bytes_per_element": logit_itemsize,
            "limits": asdict(limits), "elapsed_seconds": time.monotonic() - started,
            "overall": summary, "coverage": coverage,
            "by_level": by_level, "by_domain": by_domain,
            "time_budget_exhausted": time.monotonic() >= deadline,
            "gate": likelihood_gate(summary, coverage, len(rejected), criteria), "examples": rows,
            "recipe": {"prompt_response_bpe_boundary": "encoded_separately", "bos_in_prompt": True,
                       "eos_in_response_score": True, "nll_units": "natural_log_nats_per_response_token",
                       "aggregate_nll_weighting": "response_token_weighted",
                       "nll_interval_weighting": "resample_examples_then_compute_response_token_weighted_nll",
                       "logits_memory_cap_exclusions": "model weights, saved forward activations, and bounded float64 scoring temporaries",
                       "preference_tie_credit": 0.5, "tie_absolute_tolerance": 1e-12,
                       "generated_code_executed": False, "bootstrap_interpretation": "resampling_sensitivity_of_this_prompt_set_only",
                       "deadline_semantics": "checked_between_full_forward_calls; native NumPy kernels are not preempted"},
            "limitations": ["Teacher forcing supplies every preceding correct-response token.",
                            "Summed response probability is sensitive to answer length; average probability measures a different quantity.",
                            "These original questions are a small curated probe, not a representative programming benchmark.",
                            "Likelihood preference and perplexity do not establish freeform factual accuracy, teaching quality, or executable-code correctness."]}
