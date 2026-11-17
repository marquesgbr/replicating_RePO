"""Tests for the FPR/TPR metrics of the new dataset evaluation.

The metrics operate on the RePO/RePO+ scores produced by
:mod:`new_data_eval.scoring`; a score at or above the decision threshold is
classified as malicious, exactly like the reference notebooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from new_data_eval.metrics import (
    average_attack_tpr,
    class_rates,
    conditional_drop,
    fpr,
)


def test_class_rates_include_counts_and_low_n_flag():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array(["BENIGN", "ddos", "ddos", "mitm"])
    rates = class_rates(scores, labels, threshold=0.5,
                        expected_labels=["BENIGN", "ddos", "mitm"])
    assert rates["ddos"] == {"n": 2, "tpr": 0.5, "low_n": True}
    assert rates["mitm"]["tpr"] == 0.0


def test_average_tpr_excludes_benign_and_reports_weighted():
    rates = {
        "BENIGN": {"n": 100, "fpr": 0.01},
        "ddos": {"n": 3, "tpr": 1.0, "low_n": True},
        "xss": {"n": 1, "tpr": 0.0, "low_n": True},
    }
    averages = average_attack_tpr(rates, low_n_threshold=100)
    assert averages["macro"] == 0.5
    assert averages["weighted"] == 0.75


def test_conditional_drop_returns_none_for_zero_normal_tpr():
    assert conditional_drop(0.0, 0.0) is None


def test_fpr_is_the_fraction_of_benign_scores_at_or_above_threshold():
    scores = np.array([0.9, 0.8, 0.5, 0.49, 0.1])
    labels = np.array(["BENIGN", "ddos", "BENIGN", "BENIGN", "ddos"])
    assert fpr(scores, labels, 0.5) == pytest.approx(2 / 3)
    assert fpr(scores, labels, 0.9) == pytest.approx(1 / 3)


def test_fpr_ignores_attack_scores():
    scores = np.array([0.1, 0.9, 0.9])
    labels = np.array(["BENIGN", "ddos", "xss"])
    assert fpr(scores, labels, 0.5) == 0.0


def test_fpr_without_benign_rows_returns_zero():
    scores = np.array([0.1, 0.9])
    labels = np.array(["ddos", "xss"])
    assert fpr(scores, labels, 0.5) == 0.0


def test_class_rates_include_every_expected_label_even_when_absent():
    scores = np.array([0.9, 0.8, 0.1])
    labels = np.array(["BENIGN", "ddos", "ddos"])
    rates = class_rates(
        scores, labels, threshold=0.5, expected_labels=["BENIGN", "ddos", "xss"]
    )
    assert set(rates) == {"BENIGN", "ddos", "xss"}
    assert rates["xss"] == {"n": 0, "tpr": 0.0, "low_n": True}


def test_class_rates_benign_reports_count_fpr_and_tpr():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array(["BENIGN", "ddos", "ddos", "mitm"])
    rates = class_rates(
        scores, labels, threshold=0.5, expected_labels=["BENIGN", "ddos", "mitm"]
    )
    assert rates["BENIGN"]["n"] == 1
    assert rates["BENIGN"]["fpr"] == 1.0
    assert rates["BENIGN"]["tpr"] == 1.0
    assert "low_n" not in rates["BENIGN"]


def test_class_rates_can_mark_a_large_class_as_not_low_n():
    scores = np.concatenate([np.array([0.9, 0.1]), np.zeros(101)])
    labels = np.array(["BENIGN"] + ["ddos"] * 101 + ["mitm"])
    rates = class_rates(
        scores, labels, threshold=0.5, expected_labels=["BENIGN", "ddos", "mitm"]
    )
    assert rates["ddos"]["n"] == 101
    assert rates["ddos"]["low_n"] is False
    assert rates["mitm"]["low_n"] is True


def test_class_rates_reject_unknown_labels():
    scores = np.array([0.9, 0.1])
    labels = np.array(["BENIGN", "unknown"])
    with pytest.raises(ValueError, match="unknown"):
        class_rates(
            scores, labels, threshold=0.5, expected_labels=["BENIGN", "ddos"]
        )


def test_average_tpr_excluding_low_n_uses_only_large_classes():
    rates = {
        "BENIGN": {"n": 500, "fpr": 0.0},
        "ddos": {"n": 200, "tpr": 1.0, "low_n": False},
        "xss": {"n": 4, "tpr": 0.0, "low_n": True},
        "mitm": {"n": 300, "tpr": 0.5, "low_n": False},
    }
    averages = average_attack_tpr(rates, low_n_threshold=100)
    assert averages["macro"] == pytest.approx(0.5)
    assert averages["weighted"] == pytest.approx(
        (200 * 1.0 + 4 * 0.0 + 300 * 0.5) / 504
    )
    assert averages["macro_excluding_low_n"] == pytest.approx(0.75)


def test_average_tpr_of_no_attack_is_zero():
    averages = average_attack_tpr(
        {"BENIGN": {"n": 10, "fpr": 0.1}}, low_n_threshold=100
    )
    assert averages == {"macro": 0.0, "weighted": 0.0, "macro_excluding_low_n": 0.0}


def test_average_tpr_ignores_keys_outside_attack_labels():
    rates = {
        "BENIGN": {"n": 10, "fpr": 0.1},
        "ddos": {"n": 2, "tpr": 1.0, "low_n": True},
        "unexpected": {"n": 100, "tpr": 0.0, "low_n": False},
    }
    averages = average_attack_tpr(rates, low_n_threshold=100)
    assert averages["macro"] == pytest.approx(1.0)
    assert averages["weighted"] == pytest.approx(1.0)
    assert averages["macro_excluding_low_n"] == 0.0


def test_conditional_drop_is_relative_tpr_loss():
    assert conditional_drop(0.25, 0.5) == pytest.approx(0.5)
    assert conditional_drop(0.5, 0.5) == pytest.approx(0.0)


def test_conditional_drop_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        conditional_drop(float("nan"), 0.5)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="different lengths"):
        class_rates(
            np.array([0.1, 0.2]),
            np.array(["BENIGN"]),
            threshold=0.5,
            expected_labels=["BENIGN"],
        )
    with pytest.raises(ValueError, match="different lengths"):
        fpr(np.array([0.1, 0.2]), np.array(["BENIGN"]), 0.5)


def test_metrics_reject_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        fpr(
            np.array([0.1, float("nan")]),
            np.array(["BENIGN", "BENIGN"]),
            0.5,
        )
