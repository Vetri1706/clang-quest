"""NumPy numerical kernels, including exact tiled softmax attention.

The attention implementation follows the online-softmax recurrence and
recomputes score tiles during its backward pass. It never allocates a full
query-by-key score or probability matrix. This is an inspectable CPU
reference, not a fused accelerator kernel.
"""

from __future__ import annotations

import math
import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dispatch dense multiplication to the NumPy backend's BLAS kernel."""
    return np.matmul(a, b)


def matmul_tiled(a: np.ndarray, b: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Explicit 2-D blocked matrix multiplication with bounded temporary tiles."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul_tiled requires compatible rank-two operands")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    dtype = np.result_type(a.dtype, b.dtype)
    out = np.zeros((a.shape[0], b.shape[1]), dtype=dtype)
    for i in range(0, a.shape[0], block_size):
        for j in range(0, b.shape[1], block_size):
            tile = out[i : i + block_size, j : j + block_size]
            for k in range(0, a.shape[1], block_size):
                tile += np.matmul(a[i : i + block_size, k : k + block_size],
                                  b[k : k + block_size, j : j + block_size])
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax; an entirely -infinity row has all-zero probability.

    A row containing positive infinities distributes its probability evenly
    among those positions. NaNs are rejected instead of silently entering a
    training graph.
    """
    x = np.asarray(x)
    if x.dtype.kind != "f":
        x = x.astype(np.float64)
    if np.isnan(x).any():
        raise ValueError("softmax input contains NaN")
    work = x.astype(np.result_type(x.dtype, np.float32), copy=False)
    m = np.max(work, axis=axis, keepdims=True)
    shifted = np.full_like(work, -np.inf)
    np.subtract(work, m, out=shifted, where=np.isfinite(work) & np.isfinite(m))
    e = np.exp(shifted)
    has_pos_inf = np.isposinf(m)
    e = np.where(has_pos_inf, np.isposinf(work).astype(work.dtype), e)
    total = e.sum(axis=axis, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total != 0)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log softmax without forming log(exp(x)); stable for very small tails."""
    x = np.asarray(x)
    work = x.astype(np.result_type(x.dtype, np.float32), copy=False)
    if np.isnan(work).any():
        raise ValueError("log_softmax input contains NaN")
    m = np.max(work, axis=axis, keepdims=True)
    shifted = np.full_like(work, -np.inf)
    np.subtract(work, m, out=shifted, where=np.isfinite(work) & np.isfinite(m))
    totals = np.exp(shifted).sum(axis=axis, keepdims=True)
    logtotal = np.zeros_like(totals)
    np.log(totals, out=logtotal, where=totals > 0)
    result = shifted - logtotal
    positive = np.isposinf(m)
    if positive.any():
        counts = np.isposinf(work).sum(axis=axis, keepdims=True)
        logcounts = np.zeros_like(m)
        np.log(counts, out=logcounts, where=counts > 0)
        result = np.where(positive & np.isposinf(work), -logcounts, result)
    return result


def _validate_attention(q, k, v, mask, block_size, scale):
    q, k, v = np.asarray(q), np.asarray(k), np.asarray(v)
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("attention expects [batch, heads, tokens, features] arrays")
    if any(x.dtype.kind != "f" for x in (q, k, v)):
        raise TypeError("attention operands must be floating point")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("attention batch, heads, and key/value lengths must agree")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] < 1:
        raise ValueError("query and key features must agree and be nonempty")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("attention mask must be boolean; True means allowed")
        mask = np.broadcast_to(mask, (*q.shape[:3], k.shape[2]))
    dtype = np.result_type(q.dtype, k.dtype, v.dtype, np.float32)
    return (q.astype(dtype, copy=False), k.astype(dtype, copy=False),
            v.astype(dtype, copy=False), mask, scale)


def _scores(q, k, qi, ki, causal, mask, query_offset, scale):
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    allowed = None
    if causal:
        qpos = query_offset + np.arange(qi, qi + q.shape[2])
        kpos = np.arange(ki, ki + k.shape[2])
        allowed = kpos[None, :] <= qpos[:, None]
    if mask is not None:
        mtile = mask[:, :, qi : qi + q.shape[2], ki : ki + k.shape[2]]
        allowed = mtile if allowed is None else allowed & mtile
    return np.where(allowed, scores, -np.inf) if allowed is not None else scores


def attention_forward(q: np.ndarray, k: np.ndarray, v: np.ndarray, *,
                      causal: bool = True, mask: np.ndarray | None = None,
                      query_offset: int = 0, block_size: int = 32,
                      scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact online softmax attention; return output and per-row log normalizer.

    Working storage is O(B H (Tq Dv + Tq + block_size**2)) in addition to
    inputs and an optional caller-provided mask. Rows with no allowed keys
    produce zero output and a -infinity log normalizer.
    """
    q, k, v, mask, scale = _validate_attention(q, k, v, mask, block_size, scale)
    out = np.zeros((*q.shape[:3], v.shape[-1]), dtype=q.dtype)
    log_normalizer = np.full(q.shape[:3], -np.inf, dtype=q.dtype)
    for qi in range(0, q.shape[2], block_size):
        qt = q[:, :, qi : qi + block_size]
        m = np.full(qt.shape[:3], -np.inf, dtype=q.dtype)
        denominator = np.zeros_like(m)
        numerator = np.zeros((*qt.shape[:3], v.shape[-1]), dtype=q.dtype)
        for ki in range(0, k.shape[2], block_size):
            kt, vt = k[:, :, ki : ki + block_size], v[:, :, ki : ki + block_size]
            scores = _scores(qt, kt, qi, ki, causal, mask, query_offset, scale)
            new_m = np.maximum(m, scores.max(axis=-1))
            shift = np.full_like(m, -np.inf)
            np.subtract(m, new_m, out=shift, where=np.isfinite(m) & np.isfinite(new_m))
            alpha = np.exp(shift)
            shifted_scores = np.full_like(scores, -np.inf)
            np.subtract(scores, new_m[..., None], out=shifted_scores,
                        where=np.isfinite(scores) & np.isfinite(new_m[..., None]))
            probabilities = np.exp(shifted_scores)
            numerator = numerator * alpha[..., None] + np.matmul(probabilities, vt)
            denominator = denominator * alpha + probabilities.sum(axis=-1)
            m = new_m
        out[:, :, qi : qi + block_size] = np.divide(
            numerator, denominator[..., None], out=np.zeros_like(numerator),
            where=denominator[..., None] > 0)
        logd = np.full_like(denominator, -np.inf)
        np.log(denominator, out=logd, where=denominator > 0)
        log_normalizer[:, :, qi : qi + block_size] = m + logd
    return out, log_normalizer


def attention_backward(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                       output: np.ndarray, log_normalizer: np.ndarray,
                       grad_output: np.ndarray, *, causal: bool = True,
                       mask: np.ndarray | None = None, query_offset: int = 0,
                       block_size: int = 32, scale: float | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute attention tiles and return analytical dQ, dK, and dV."""
    q, k, v, mask, scale = _validate_attention(q, k, v, mask, block_size, scale)
    if output.shape != (*q.shape[:3], v.shape[-1]) or grad_output.shape != output.shape:
        raise ValueError("attention output/gradient shape mismatch")
    if log_normalizer.shape != q.shape[:3]:
        raise ValueError("attention log normalizer shape mismatch")
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    grad_output = np.asarray(grad_output, dtype=q.dtype)
    row_delta = np.sum(grad_output * output, axis=-1)
    for qi in range(0, q.shape[2], block_size):
        qt, gt = q[:, :, qi : qi + block_size], grad_output[:, :, qi : qi + block_size]
        lse = log_normalizer[:, :, qi : qi + block_size, None]
        delta = row_delta[:, :, qi : qi + block_size, None]
        for ki in range(0, k.shape[2], block_size):
            kt, vt = k[:, :, ki : ki + block_size], v[:, :, ki : ki + block_size]
            scores = _scores(qt, kt, qi, ki, causal, mask, query_offset, scale)
            shifted = np.full_like(scores, -np.inf)
            np.subtract(scores, lse, out=shifted, where=np.isfinite(scores) & np.isfinite(lse))
            p = np.exp(shifted)
            dp = np.matmul(gt, vt.swapaxes(-1, -2))
            ds = p * (dp - delta)
            dq[:, :, qi : qi + block_size] += np.matmul(ds, kt) * scale
            dk[:, :, ki : ki + block_size] += np.matmul(ds.swapaxes(-1, -2), qt) * scale
            dv[:, :, ki : ki + block_size] += np.matmul(p.swapaxes(-1, -2), gt)
    return dq, dk, dv
