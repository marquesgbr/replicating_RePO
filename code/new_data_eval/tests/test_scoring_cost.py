"""Tests for the scoring cost counters.

The counters are instrumentation only: they must describe the work performed
(rows sent to the model and number of forward passes) without ever changing a
score.
"""

from __future__ import annotations

import numpy as np

from new_data_eval.scoring import (
    ScoringCost,
    reconstruction_errors,
    score_samples,
)


class CountingModel:
    """Identity model that counts its forward passes."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs, training=False):
        del training
        self.calls += 1
        return np.asarray(inputs, dtype=np.float32)


def test_cost_counts_rows_and_forward_passes(meta):
    model = CountingModel()
    cost = ScoringCost()
    x = np.zeros((3, 77), dtype=np.float32)

    reconstruction_errors(
        model, x, *meta, n_masks=10, batch_size=4, cost=cost
    )

    # 3 samples x 10 masks = 30 masked rows; with a batch of 4 they are sent
    # in ceil(30 / 4) = 8 forward passes.
    assert cost.masked_rows == 30
    assert cost.model_calls == 8
    assert cost.model_calls == model.calls
    assert cost.model_seconds >= 0.0
    assert len(cost.per_call_seconds) == 8

    payload = cost.as_dict()
    assert set(payload) == {
        "masked_rows",
        "model_calls",
        "model_seconds",
        "mean_call_ms",
    }
    assert payload["masked_rows"] == 30
    assert payload["model_calls"] == 8
    assert payload["mean_call_ms"] >= 0.0


def test_cost_multiplies_by_score_repeats(meta):
    cost = ScoringCost()
    x = np.zeros((2, 77), dtype=np.float32)
    reconstruction_errors(
        CountingModel(),
        x,
        *meta,
        n_masks=10,
        batch_size=100,
        score_repeats=3,
        cost=cost,
    )
    assert cost.masked_rows == 2 * 10 * 3
    assert cost.model_calls == 3  # one full batch per repeat


def test_cost_does_not_change_a_score(meta):
    x = np.random.default_rng(0).random((5, 77)).astype(np.float32)
    kwargs = dict(mode="repo_plus", n_masks=20, seed=2019, batch_size=7)

    plain = score_samples(CountingModel(), x, *meta, **kwargs)
    cost = ScoringCost()
    instrumented = score_samples(CountingModel(), x, *meta, cost=cost, **kwargs)

    np.testing.assert_allclose(instrumented, plain, rtol=0.0, atol=0.0)
    assert cost.masked_rows == 5 * 20


def test_cost_merge_and_empty_dict():
    empty = ScoringCost().as_dict()
    assert empty == {
        "masked_rows": 0,
        "model_calls": 0,
        "model_seconds": 0.0,
        "mean_call_ms": 0.0,
    }

    first = ScoringCost()
    first.add_call(10, 0.25)
    second = ScoringCost()
    second.add_call(5, 0.75)
    first.merge(second)

    assert first.masked_rows == 15
    assert first.model_calls == 2
    assert first.model_seconds == 1.0
    assert first.as_dict()["mean_call_ms"] == 500.0
