"""Cheap decision rules on top of the per-mask reconstruction errors.

Two strategies that keep the RePO+ detector but spend fewer model calls:

* :func:`cascade_decisions` -- a cheap prefix of the masks (one or a few per
  RePO+ group) is scored first; samples whose cheap score is already below a
  low threshold ``t_low`` are declared benign without the full ensemble.  The
  rest pay the full 100-mask score.
* :func:`sequential_repo_decisions` -- for the mean-based RePO score, masks are
  drawn in blocks and the decision stops as soon as the confidence interval of
  the mean excludes the threshold.

Both functions are pure NumPy and operate on a pre-computed
``(n_samples, n_masks)`` error matrix, so they can be evaluated without
re-running the model (the cost of the rule is the number of masks it *would*
use, reported per sample).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from new_data_eval.masking import REPO_PLUS_GROUPS

__all__ = [
    "CascadeResult",
    "SequentialResult",
    "cascade_decisions",
    "cascade_threshold",
    "cheap_repo_plus",
    "sequential_repo_decisions",
]


@dataclass(frozen=True)
class CascadeResult:
    """Decision of the cascade and the number of masks it used."""

    score: np.ndarray
    routed: np.ndarray
    masks_used: np.ndarray
    decision: np.ndarray


@dataclass(frozen=True)
class SequentialResult:
    """Decision of the sequential RePO rule."""

    decision: np.ndarray
    masks_used: np.ndarray


def cheap_repo_plus(errors: np.ndarray, n_cheap: int) -> np.ndarray:
    """RePO+ score computed from the first ``n_cheap`` masks.

    ``n_cheap`` must be a multiple of :data:`REPO_PLUS_GROUPS`; the prefix is
    split into the same five groups and the minima are summed.
    """
    array = np.asarray(errors, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"errors must be 2-D, got shape {array.shape}")
    if n_cheap <= 0 or n_cheap % REPO_PLUS_GROUPS != 0:
        raise ValueError(
            f"n_cheap must be a positive multiple of {REPO_PLUS_GROUPS}, "
            f"got {n_cheap}"
        )
    if n_cheap > array.shape[1]:
        raise ValueError(
            f"n_cheap ({n_cheap}) exceeds the number of masks ({array.shape[1]})"
        )
    grouped = array[:, :n_cheap].reshape(
        array.shape[0], REPO_PLUS_GROUPS, n_cheap // REPO_PLUS_GROUPS
    )
    return grouped.min(axis=2).sum(axis=1)


def cascade_threshold(
    attack_cheap: np.ndarray,
    *,
    false_negative_rate: float = 0.0,
) -> float:
    """Return the largest ``t_low`` that keeps the false negative rate bounded.

    A sample is declared benign by the cheap stage when its cheap score is
    below ``t_low``; the false negative rate is the fraction of (attack)
    samples that the cheap stage would discard.  With the default of zero the
    threshold is just below the smallest attack score observed.
    """
    values = np.asarray(attack_cheap, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("attack_cheap must contain at least one score")
    if not 0.0 <= false_negative_rate < 1.0:
        raise ValueError(
            f"false_negative_rate must be in [0, 1), got {false_negative_rate!r}"
        )
    if false_negative_rate == 0.0:
        return float(values.min()) - 1e-12
    return float(np.quantile(values, false_negative_rate))


def cascade_decisions(
    errors: np.ndarray,
    *,
    n_cheap: int,
    t_low: float,
    threshold: float,
) -> CascadeResult:
    """Apply the cheap-first cascade to a pre-computed error matrix.

    ``threshold`` is the decision threshold of the *final* score.  Non-routed
    samples use the cheap score; routed samples use the full RePO+ score.
    """
    array = np.asarray(errors, dtype=np.float64)
    cheap = cheap_repo_plus(array, n_cheap)
    routed = cheap >= t_low
    score = cheap.copy()
    if routed.any():
        full = cheap_repo_plus(array[routed], array.shape[1])
        score[routed] = full
    masks_used = np.where(routed, array.shape[1], n_cheap)
    decision = score >= threshold
    return CascadeResult(
        score=score, routed=routed, masks_used=masks_used, decision=decision
    )


def sequential_repo_decisions(
    errors: np.ndarray,
    *,
    threshold: float,
    z: float = 1.96,
    block: int = 5,
    min_masks: int = 5,
    max_masks: int | None = None,
) -> SequentialResult:
    """Decide the mean-based RePO score with a confidence-interval rule.

    Masks are revealed in blocks; after each block the 95 % interval of the
    mean is computed.  If the whole interval is above the threshold the sample
    is declared malicious; if it is entirely below, benign; otherwise the next
    block is drawn.  At ``max_masks`` the point estimate decides.
    """
    array = np.asarray(errors, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"errors must be 2-D, got shape {array.shape}")
    total = array.shape[1]
    max_masks = total if max_masks is None else min(int(max_masks), total)
    if block <= 0 or min_masks <= 0:
        raise ValueError("block and min_masks must be positive")

    n_samples = array.shape[0]
    decision = np.zeros(n_samples, dtype=bool)
    masks_used = np.full(n_samples, max_masks, dtype=np.int64)
    pending = np.ones(n_samples, dtype=bool)

    for count in range(min_masks, max_masks + 1, block):
        rows = np.flatnonzero(pending)
        if rows.size == 0:
            break
        sample = array[rows, :count]
        mean = sample.mean(axis=1)
        if count > 1:
            standard_error = sample.std(axis=1, ddof=1) / np.sqrt(count)
        else:
            standard_error = np.zeros(rows.size)
        low = mean - z * standard_error
        high = mean + z * standard_error

        decided_high = low >= threshold
        decided_low = high < threshold
        decided = decided_high | decided_low
        decision[rows[decided_high]] = True
        decision[rows[decided_low]] = False
        masks_used[rows[decided]] = count
        pending[rows[decided]] = False

    if pending.any():
        rows = np.flatnonzero(pending)
        mean = array[rows, :max_masks].mean(axis=1)
        decision[rows] = mean >= threshold
        masks_used[rows] = max_masks

    return SequentialResult(decision=decision, masks_used=masks_used)
