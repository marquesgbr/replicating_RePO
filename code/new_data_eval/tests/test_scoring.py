"""Tests for the RePO/RePO+ scoring primitives and the threshold calibration.

The expectations mirror the reference implementation of the Apuana runner
(``cluster/metrics.py``) so the new dataset evaluation uses the very same
protocol as the flow/packet replications.
"""

from __future__ import annotations

import numpy as np
import pytest

from new_data_eval.scoring import (
    PHASE_ATTACK,
    PHASE_ORIGINAL,
    PHASE_VICTIM,
    repo_plus_scores,
    repo_scores,
    score_from_errors,
    threshold_at_fpr,
)


def test_repo_score_is_mean_of_mask_errors():
    errors = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(repo_scores(errors), [2.0, 3.0])


def test_repo_plus_uses_five_groups_of_twenty():
    errors = np.arange(100, dtype=float).reshape(1, 100)
    expected = sum(np.min(errors.reshape(1, 5, 20), axis=2).ravel())
    assert repo_plus_scores(errors)[0] == expected


def test_threshold_at_fpr_uses_ceil_and_is_deterministic_with_ties():
    scores = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    assert threshold_at_fpr(scores, 0.3) == 3.0
    assert threshold_at_fpr(scores, 0.3) == threshold_at_fpr(scores, 0.3)


def test_threshold_at_fpr_rejects_empty_benign():
    with pytest.raises(ValueError, match="empty"):
        threshold_at_fpr(np.array([]), 0.01)


def test_repo_scores_accepts_a_single_sample_row():
    np.testing.assert_allclose(repo_scores(np.array([1.0, 3.0])), [2.0])


def test_repo_plus_rejects_mask_count_not_divisible_by_five():
    errors = np.arange(7, dtype=float).reshape(1, 7)
    with pytest.raises(ValueError, match="not divisible"):
        repo_plus_scores(errors)


def test_repo_plus_matches_the_reference_aggregation():
    rng = np.random.default_rng(7)
    errors = rng.random((4, 100))
    expected = np.min(errors.reshape(4, 5, 20), axis=2).sum(axis=1)
    np.testing.assert_allclose(repo_plus_scores(errors), expected)


def test_score_from_errors_dispatches_on_mode():
    errors = np.arange(100, dtype=float).reshape(1, 100)
    np.testing.assert_allclose(score_from_errors(errors, "repo"), repo_scores(errors))
    np.testing.assert_allclose(
        score_from_errors(errors, "repo_plus"), repo_plus_scores(errors)
    )


def test_score_from_errors_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown score mode"):
        score_from_errors(np.zeros((1, 10)), "not-a-mode")


def test_phase_constants_are_stable():
    assert PHASE_ORIGINAL == "original"
    assert PHASE_ATTACK == "attack"
    assert PHASE_VICTIM == "victim"


def test_threshold_at_fpr_rejects_out_of_range_fpr():
    with pytest.raises(ValueError, match="fpr"):
        threshold_at_fpr(np.array([1.0, 2.0]), 0.0)


def test_threshold_at_fpr_matches_the_reference_ceil_index():
    scores = np.arange(100, dtype=float)
    # ceil(100 * 0.01) == 1 -> the largest benign score.
    assert threshold_at_fpr(scores, 0.01) == 99.0
    # ceil(100 * 0.10) == 10 -> the 10th largest benign score.
    assert threshold_at_fpr(scores, 0.10) == 90.0


def test_threshold_at_fpr_clamps_the_index_to_the_available_scores():
    scores = np.array([1.0, 2.0, 3.0])
    # ceil(3 * 0.9) == 3 -> the smallest score.
    assert threshold_at_fpr(scores, 0.9) == 1.0

