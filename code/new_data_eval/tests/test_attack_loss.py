"""Tests for the attack objectives of :mod:`new_data_eval.attack`."""

from __future__ import annotations

import numpy as np
import pytest

from new_data_eval.attack import ci_objective, validate_objective
from new_data_eval.decision import mean_confidence_bounds


def test_validate_objective_rejects_an_unknown_loss():
    with pytest.raises(ValueError, match="loss"):
        validate_objective("median", 5)


def test_validate_objective_rejects_a_ci_objective_without_a_spread():
    with pytest.raises(ValueError, match="at least 2 masks"):
        validate_objective("ci", 1)
    validate_objective("sum", 1)
    validate_objective("ci", 5)


def test_ci_objective_matches_the_rule_upper_bound():
    """The attack criterion is the rule's own upper bound, scaled by ``n``."""
    tf = pytest.importorskip("tensorflow")

    values = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    n = values.size
    _, high = mean_confidence_bounds(values[None, :], z=1.96)

    got = float(ci_objective(tf.constant(values, dtype=tf.float32), 1.96).numpy())

    assert got == pytest.approx(float(n * high[0]), rel=1e-5)


def test_ci_objective_matches_the_point_sum_without_spread():
    tf = pytest.importorskip("tensorflow")

    values = np.full(5, 3.0)
    got = float(ci_objective(tf.constant(values, dtype=tf.float32), 1.96).numpy())

    assert got == pytest.approx(15.0)
