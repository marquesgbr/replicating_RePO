"""FPR/TPR metrics for the new dataset evaluation.

The metrics mirror the decision rule of the reference notebooks: a score at
or above the decision threshold is classified as malicious.  ``BENIGN`` is
the negative class (its rate is the false positive rate), every other class is
an attack (its rate is the true positive rate).

The module is NumPy-only and never touches TensorFlow or the attack logic.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from new_data_eval.data import BENIGN_LABEL

__all__ = [
    "average_attack_tpr",
    "class_rates",
    "conditional_drop",
    "fpr",
]


#: Classes with fewer samples than this are flagged with ``low_n`` (D12).
_LOW_N_THRESHOLD = 100


def _as_score_label_vectors(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the score/label vectors, validating their shape and finiteness."""
    score_array = np.asarray(scores, dtype=np.float64).ravel()
    label_array = np.asarray(labels).ravel()
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            "scores and labels have different lengths: "
            f"{score_array.shape[0]} != {label_array.shape[0]}"
        )
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    return score_array, label_array


def _as_threshold(threshold: float) -> float:
    """Return ``threshold`` as a finite float."""
    value = float(threshold)
    if not np.isfinite(value):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    return value


def _rate_at_threshold(scores: np.ndarray, threshold: float) -> float:
    """Fraction of ``scores`` at or above ``threshold`` (0.0 when empty)."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if values.size == 0:
        return 0.0
    return float(np.mean(values >= _as_threshold(threshold)))


def fpr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Return the false positive rate of ``threshold``.

    The FPR is the fraction of ``BENIGN`` samples whose score is at or above
    ``threshold``.  Attack scores are ignored; an input without benign rows
    yields ``0.0``.
    """
    score_array, label_array = _as_score_label_vectors(scores, labels)
    threshold = _as_threshold(threshold)
    mask = label_array == BENIGN_LABEL
    if not mask.any():
        return 0.0
    return float(np.mean(score_array[mask] >= threshold))


def class_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    expected_labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return per-class rates at ``threshold``.

    Every label of ``expected_labels`` is present in the result, even when it
    has no sample.  ``BENIGN`` reports its count, FPR and TPR (the TPR of the
    negative class is the FPR); an attack reports its count, TPR and a
    ``low_n`` flag (true when ``n < 100``).
    """
    score_array, label_array = _as_score_label_vectors(scores, labels)
    threshold = _as_threshold(threshold)

    expected = [str(label) for label in expected_labels]
    unknown = sorted({str(label) for label in label_array} - set(expected))
    if unknown:
        raise ValueError(
            f"labels contain classes outside expected_labels: {unknown}"
        )

    rates: dict[str, dict[str, Any]] = {}
    for name in expected:
        mask = label_array == name
        count = int(np.count_nonzero(mask))
        rate = _rate_at_threshold(score_array[mask], threshold) if count else 0.0
        if name == BENIGN_LABEL:
            rates[name] = {"n": count, "tpr": rate, "fpr": rate}
        else:
            rates[name] = {
                "n": count,
                "tpr": rate,
                "low_n": count < _LOW_N_THRESHOLD,
            }
    return rates


def average_attack_tpr(
    rates: Mapping[str, Mapping[str, Any]], *, low_n_threshold: int = 100
) -> dict[str, float]:
    """Return macro, weighted and low-n-excluded mean TPR of the attacks.

    ``BENIGN`` is always excluded.  ``macro`` is the simple mean of the attack
    TPRs, ``weighted`` weights each TPR by its ``n`` and
    ``macro_excluding_low_n`` averages only the classes with
    ``n >= low_n_threshold``.  An empty selection yields ``0.0`` for the
    corresponding average.
    """
    attacks = [
        (str(name), item)
        for name, item in rates.items()
        if str(name) != BENIGN_LABEL and "tpr" in item
    ]
    if not attacks:
        return {"macro": 0.0, "weighted": 0.0, "macro_excluding_low_n": 0.0}

    tprs = [float(item["tpr"]) for _, item in attacks]
    counts = [int(item.get("n", 0)) for _, item in attacks]
    total = sum(counts)

    macro = float(np.mean(tprs))
    weighted = (
        float(sum(tpr * n for tpr, n in zip(tprs, counts)) / total)
        if total
        else 0.0
    )
    large = [tpr for tpr, n in zip(tprs, counts) if n >= low_n_threshold]
    macro_excluding_low_n = float(np.mean(large)) if large else 0.0

    return {
        "macro": macro,
        "weighted": weighted,
        "macro_excluding_low_n": macro_excluding_low_n,
    }


def conditional_drop(victim_tpr: float, normal_tpr: float) -> float | None:
    """Return ``1 - victim_tpr / normal_tpr``.

    ``None`` is returned when the normal TPR is zero, because the relative
    drop is undefined in that case (there is no detection to lose).
    """
    victim = float(victim_tpr)
    normal = float(normal_tpr)
    if not np.isfinite(victim) or not np.isfinite(normal):
        raise ValueError(
            "victim_tpr and normal_tpr must be finite, got "
            f"{victim_tpr!r} and {normal_tpr!r}"
        )
    if normal == 0.0:
        return None
    return 1.0 - victim / normal
