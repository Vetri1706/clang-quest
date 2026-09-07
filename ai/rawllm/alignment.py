"""Explicit response-masked supervised learning and direct preference optimization."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, log_softmax, softplus


def sequence_log_probs(logits: Tensor, targets: np.ndarray,
                       loss_mask: np.ndarray) -> Tensor:
    """Sum target log-probabilities per sequence, preserving policy gradients."""
    targets = np.asarray(targets)
    mask = np.asarray(loss_mask, dtype=np.float32)
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or mask.shape != targets.shape:
        raise ValueError("Expected logits [B,T,V] and targets/loss_mask [B,T]")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("Targets must be integer token IDs")
    if np.any(targets < 0) or np.any(targets >= logits.shape[-1]):
        raise ValueError("Target token ID outside logits vocabulary")
    if not np.all(np.isfinite(mask)) or np.any(mask < 0):
        raise ValueError("Loss masks must be finite non-negative weights")
    batch, length = targets.shape
    selected = log_softmax(logits, axis=-1)[np.arange(batch)[:, None],
                                           np.arange(length)[None, :], targets]
    return (selected * Tensor(mask)).sum(axis=1)


def sft_loss(logits: Tensor, targets: np.ndarray, loss_mask: np.ndarray) -> Tensor:
    count = float(np.asarray(loss_mask).sum())
    if not np.isfinite(count) or count <= 0:
        raise ValueError("SFT requires at least one finite positive supervised target")
    return -sequence_log_probs(logits, targets, loss_mask).sum() / count


def dpo_loss(policy_chosen: Tensor, policy_rejected: Tensor,
             reference_chosen: Tensor | np.ndarray, reference_rejected: Tensor | np.ndarray,
             beta: float = 0.1) -> Tensor:
    """Mean -log sigmoid(beta * (policy log-odds - frozen reference log-odds)).

    Arguments are response-only sequence log-probabilities, summed rather than
    length-normalized. References are detached even if passed trainable tensors.
    Stable softplus avoids direct exponentiation of a large negative margin.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    chosen_ref = reference_chosen.detach() if isinstance(reference_chosen, Tensor) else Tensor(np.asarray(reference_chosen))
    rejected_ref = reference_rejected.detach() if isinstance(reference_rejected, Tensor) else Tensor(np.asarray(reference_rejected))
    if not (policy_chosen.shape == policy_rejected.shape == chosen_ref.shape == rejected_ref.shape):
        raise ValueError("All DPO sequence score shapes must match")
    if policy_chosen.data.size == 0:
        raise ValueError("DPO requires at least one preference pair")
    margin = beta * ((policy_chosen - policy_rejected) - (chosen_ref - rejected_ref))
    return softplus(-margin).mean()
