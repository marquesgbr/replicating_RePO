"""Tests for the adversarial evaluation of the sequential rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from new_data_eval.decision import SequentialResult
from mask_experiments.sequential_attack import (
    sequential_attack_metrics,
    validate_blocks,
)


def _result(decision, masks_used):
    return SequentialResult(
        decision=np.array(decision, dtype=bool),
        masks_used=np.array(masks_used, dtype=np.int64),
    )


def test_validate_blocks_rejects_a_block_that_does_not_divide_the_budget():
    with pytest.raises(ValueError, match="divide"):
        validate_blocks((5, 7), 100)


def test_validate_blocks_accepts_the_study_and_the_whole_budget():
    assert validate_blocks((5, 10, 25), 100) == (5, 10, 25)
    assert validate_blocks((100,), 100) == (100,)


def test_validate_blocks_rejects_an_empty_or_negative_block():
    with pytest.raises(ValueError, match="empty"):
        validate_blocks((), 100)
    with pytest.raises(ValueError, match="positive"):
        validate_blocks((0,), 100)
    with pytest.raises(ValueError, match="exceeds"):
        validate_blocks((20,), 10)


def test_metrics_report_the_early_stop_and_the_paired_decisions():
    normal = _result([True, True, False], [5, 10, 5])
    victim = _result([False, True, False], [5, 15, 25])
    metrics = sequential_attack_metrics(
        normal=normal,
        victim=victim,
        rows=np.array([0, 1, 2]),
        block=5,
        transfer_tpr=0.5,
        repo_plus_tpr=0.4,
    )
    assert metrics["n"] == 3
    assert metrics["normal_tpr"] == pytest.approx(2 / 3)
    assert metrics["victim_tpr"] == pytest.approx(1 / 3)
    assert metrics["fooled"] == 1
    assert metrics["already_benign"] == 1
    assert metrics["early_stop_rate"] == pytest.approx(1 / 3)
    assert metrics["victim_masks_used"] == pytest.approx(15.0)
    assert metrics["conditional_drop"] == pytest.approx(0.5)
    assert metrics["transfer_full100_tpr"] == 0.5
    assert metrics["repo_plus_full100_tpr"] == 0.4
    json.dumps(metrics, allow_nan=False)


def test_metrics_are_json_safe_when_nothing_is_detected():
    empty = _result([False, False], [100, 100])
    metrics = sequential_attack_metrics(
        normal=empty,
        victim=empty,
        rows=np.array([0, 1]),
        block=5,
        transfer_tpr=0.0,
        repo_plus_tpr=0.0,
    )
    assert metrics["conditional_drop"] is None
    assert metrics["early_stop_rate"] == 0.0
    assert metrics["fooled"] == 0
    json.dumps(metrics, allow_nan=False)


def test_metrics_use_only_the_selected_rows():
    normal = _result([True, False, True], [5, 5, 5])
    victim = _result([False, False, False], [5, 5, 5])
    metrics = sequential_attack_metrics(
        normal=normal,
        victim=victim,
        rows=np.array([0, 2]),
        block=5,
        transfer_tpr=0.0,
        repo_plus_tpr=0.0,
    )
    assert metrics["n"] == 2
    assert metrics["normal_tpr"] == pytest.approx(1.0)


def test_restrict_result_aligns_the_normal_decisions_with_the_victims():
    from mask_experiments.sequential_attack import restrict_result

    normal = _result([True, False, True, False], [5, 5, 10, 25])
    restricted = restrict_result(normal, np.array([1, 3]))

    assert restricted.decision.tolist() == [False, False]
    assert restricted.masks_used.tolist() == [5, 25]
    # The caller's result is untouched.
    assert normal.decision.tolist() == [True, False, True, False]


def test_repo_plus_is_five_times_the_mean_only_for_five_masks():
    """Why the block-5 study needed no score-mode fix, and 10/25 did."""
    from new_data_eval.scoring import repo_plus_scores

    errors = np.array([[0.1, 0.2, 0.3, 0.4, 0.9], [1.0, 2.0, 3.0, 4.0, 5.0]])
    np.testing.assert_allclose(
        repo_plus_scores(errors), 5.0 * errors.mean(axis=1)
    )

    wider = np.repeat(errors, 2, axis=1)  # 10 masks, groups of 2
    assert not np.allclose(repo_plus_scores(wider), 10.0 * wider.mean(axis=1))
