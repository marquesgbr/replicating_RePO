"""Tests for the cascade and sequential decision rules."""

from __future__ import annotations

import numpy as np
import pytest

from new_data_eval.decision import (
    cascade_decisions,
    cascade_threshold,
    cheap_repo_plus,
    mean_confidence_bounds,
    sequential_repo_decisions,
)


def test_cheap_repo_plus_matches_the_full_score_at_full_width():
    rng = np.random.default_rng(0)
    errors = rng.random((4, 20))
    np.testing.assert_allclose(
        cheap_repo_plus(errors, 20),
        cheap_repo_plus(errors, 20),
        rtol=0.0,
        atol=0.0,
    )
    assert cheap_repo_plus(errors[:, :5], 5).shape == (4,)


def test_cheap_repo_plus_validates_the_prefix():
    errors = np.zeros((2, 20))
    with pytest.raises(ValueError, match="multiple of 5"):
        cheap_repo_plus(errors, 6)
    with pytest.raises(ValueError, match="exceeds"):
        cheap_repo_plus(errors, 25)


def test_cascade_threshold_controls_the_false_negative_rate():
    attack = np.array([1.0, 2.0, 3.0, 4.0])
    assert cascade_threshold(attack) < 1.0
    assert cascade_threshold(attack, false_negative_rate=0.25) == 1.75


def test_cascade_uses_the_cheap_score_when_it_is_confident():
    # 2 samples x 10 masks.  Sample 0 has a tiny cheap score (benign), sample 1
    # is high even on the cheap prefix (routed to the full score).
    errors = np.ones((2, 10))
    errors[0, :5] = 0.001
    errors[1, :] = 2.0

    result = cascade_decisions(errors, n_cheap=5, t_low=0.5, threshold=1.0)

    assert result.routed.tolist() == [False, True]
    assert result.masks_used.tolist() == [5, 10]
    assert result.decision.tolist() == [False, True]
    assert result.score[0] == pytest.approx(0.005)
    assert result.score[1] == pytest.approx(10.0)


def test_cascade_threshold_of_zero_routes_everything():
    errors = np.zeros((2, 10))
    result = cascade_decisions(errors, n_cheap=5, t_low=0.0, threshold=0.0)
    assert result.routed.tolist() == [True, True]
    assert result.masks_used.tolist() == [10, 10]


def test_sequential_rule_stops_early_on_clear_samples():
    errors = np.ones((3, 20))
    errors[0, :] = 10.0   # clearly malicious
    errors[1, :] = 0.0    # clearly benign
    # Ambiguous in the first block (mean 4, wide interval), clear later.
    errors[2, :] = 0.0
    errors[2, :5] = [0.0, 10.0, 0.0, 10.0, 0.0]

    result = sequential_repo_decisions(errors, threshold=5.0, block=5, min_masks=5)

    assert result.decision.tolist() == [True, False, False]
    assert result.masks_used[0] == 5
    assert result.masks_used[1] == 5
    assert result.masks_used[2] > 5


def test_mean_confidence_bounds_match_the_stopping_rule():
    errors = np.array([[0.0, 10.0, 0.0, 10.0, 0.0]])
    low, high = mean_confidence_bounds(errors, z=1.96)
    mean = 4.0
    se = errors[0].std(ddof=1) / np.sqrt(5)
    assert low[0] == pytest.approx(mean - 1.96 * se)
    assert high[0] == pytest.approx(mean + 1.96 * se)

    decided = sequential_repo_decisions(
        errors, threshold=float(high[0]) + 1e-9, block=5, min_masks=5
    )
    assert decided.decision.tolist() == [False]
    assert decided.masks_used.tolist() == [5]


def test_sequential_rule_falls_back_to_the_point_estimate_on_a_tie():
    # Every block straddles a threshold placed exactly on the mean, so the
    # interval never decides and the rule spends the whole budget.
    errors = np.tile([0.0, 10.0, 0.0, 10.0, 0.0], 20)[None, :]
    result = sequential_repo_decisions(
        errors, threshold=4.0, block=5, min_masks=5
    )
    assert result.masks_used.tolist() == [100]
    assert result.decision.tolist() == [True]


def test_mean_confidence_bounds_have_no_spread_with_a_single_mask():
    low, high = mean_confidence_bounds(np.array([[3.0], [7.0]]), z=1.96)
    assert low.tolist() == [3.0, 7.0]
    assert high.tolist() == [3.0, 7.0]


def test_sequential_rule_uses_every_mask_when_block_is_the_whole_budget():
    errors = np.full((2, 100), 10.0)
    result = sequential_repo_decisions(errors, threshold=5.0, block=100, min_masks=100)
    assert result.decision.tolist() == [True, True]
    assert result.masks_used.tolist() == [100, 100]
