"""RePO/RePO+ scoring primitives and threshold calibration.

The scoring follows the reference implementation of the Apuana runner
(``cluster/metrics.py``) so the results of the new dataset evaluation use the
very same protocol as the flow/packet replications:

* **RePO** -- the mean of the per-mask reconstruction errors;
* **RePO+** -- the per-mask errors are split into five groups, the minimum of
  every group is taken and the five minima are summed (100 masks -> five
  groups of 20);
* the decision threshold is the ``ceil(n * fpr)``-th largest benign score,
  counted from the end of the sorted benign scores.

Everything in this module is NumPy-only: TensorFlow is not needed to
aggregate the per-mask errors produced by the model scoring, and the module
must stay importable on a machine without a deep learning stack.
"""

from __future__ import annotations

import math

import numpy as np

from new_data_eval.data import N_FEATURES

__all__ = [
    "N_FEATURES",
    "PHASES",
    "PHASE_ATTACK",
    "PHASE_ORIGINAL",
    "PHASE_VICTIM",
    "REPO_PLUS_GROUPS",
    "per_mask_mse",
    "repo_plus_scores",
    "repo_scores",
    "score_from_errors",
    "threshold_at_fpr",
]


#: Phase of the score computed on the untouched (original) sample.
PHASE_ORIGINAL = "original"

#: Phase of the score computed during the attack optimization.
PHASE_ATTACK = "attack"

#: Phase of the score computed on the adversarial (victim) sample.
PHASE_VICTIM = "victim"

#: Every phase used to derive the stateless mask seeds.
PHASES: tuple[str, ...] = (PHASE_ORIGINAL, PHASE_ATTACK, PHASE_VICTIM)

#: Number of groups used by the RePO+ aggregate (100 masks -> 5 groups of 20).
REPO_PLUS_GROUPS = 5


def _as_error_matrix(errors: np.ndarray) -> np.ndarray:
    """Return ``errors`` as a finite ``(n_samples, n_masks)`` float64 matrix."""
    array = np.asarray(errors, dtype=np.float64)
    if array.ndim == 1:
        # A single sample: one row of per-mask errors.
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"errors must be 2-D, got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError("errors must contain at least one mask")
    if not np.isfinite(array).all():
        raise ValueError("errors must be finite")
    return array


def per_mask_mse(residuals: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error of every ``(sample, mask)`` pair.

    ``residuals`` has shape ``(n_samples, n_masks, N_FEATURES)`` -- the
    difference between the reconstruction and the original sample for every
    mask -- and the result has shape ``(n_samples, n_masks)``.

    This reproduces ``tf.reduce_mean(tf.square(rec_x - x), axis=1)`` of the
    reference notebooks without importing TensorFlow.
    """
    array = np.asarray(residuals, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"residuals must be 3-D, got shape {array.shape}")
    if array.shape[2] != N_FEATURES:
        raise ValueError(
            f"residuals must have {N_FEATURES} features in the last axis, "
            f"got shape {array.shape}"
        )
    return np.mean(np.square(array), axis=2)


def repo_scores(errors: np.ndarray) -> np.ndarray:
    """Return the RePO score of every sample (the mean of its mask errors).

    ``errors`` has shape ``(n_samples, n_masks)``; the result has shape
    ``(n_samples,)``.
    """
    return _as_error_matrix(errors).mean(axis=1)


def repo_plus_scores(errors: np.ndarray) -> np.ndarray:
    """Return the RePO+ score of every sample.

    The masks are split into :data:`REPO_PLUS_GROUPS` equally sized groups,
    the minimum error of every group is taken and the minima are summed
    (100 masks -> five groups of 20).  The number of masks must be divisible
    by the number of groups, otherwise the split is ambiguous and a
    :class:`ValueError` is raised.
    """
    array = _as_error_matrix(errors)
    n_masks = array.shape[1]
    if n_masks % REPO_PLUS_GROUPS != 0:
        raise ValueError(
            f"number of masks ({n_masks}) is not divisible by groups "
            f"({REPO_PLUS_GROUPS})"
        )
    grouped = array.reshape(array.shape[0], REPO_PLUS_GROUPS, -1)
    return grouped.min(axis=2).sum(axis=1)


def score_from_errors(errors: np.ndarray, mode: str) -> np.ndarray:
    """Return the scores of ``errors`` for the requested ``mode``.

    ``mode="repo"`` returns the RePO scores and ``mode="repo_plus"`` the
    RePO+ scores; every other mode raises a :class:`ValueError`.
    """
    if mode == "repo":
        return repo_scores(errors)
    if mode in ("repo_plus", "plus"):
        return repo_plus_scores(errors)
    raise ValueError(f"unknown score mode: {mode!r}")


def threshold_at_fpr(
    benign_scores: np.ndarray, fpr: float, method: str = "ceil"
) -> float:
    """Return the decision threshold of the benign scores at ``fpr``.

    The threshold is the ``ceil(n * fpr)``-th largest benign score (``method
    "ceil"``), counted from the end of the sorted scores -- the same rule the
    flow notebook and the Apuana runner use.  The ``"round"`` method (used by
    the packet notebook) is supported as well.  The index is clamped to
    ``[1, n]``, so the threshold always exists and is one of the benign
    scores.  Non-finite benign scores are rejected: they would otherwise
    silently poison the threshold.
    """
    values = np.asarray(benign_scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot compute a threshold from an empty score array")
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr!r}")
    if not np.isfinite(values).all():
        raise ValueError("benign scores must be finite")

    if method == "ceil":
        index = int(math.ceil(values.size * fpr))
    elif method == "round":
        index = int(np.round(values.size * fpr))
    else:
        raise ValueError(f"unknown threshold method: {method!r}")

    index = max(1, min(index, values.size))
    return float(np.sort(values)[-index])
