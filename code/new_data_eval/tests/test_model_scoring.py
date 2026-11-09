"""Tests for the model loading and the mask based RePO/RePO+ scoring.

The protocol implemented here is the one of the reference notebooks: every
sample receives 100 independent binary masks (each feature is masked with
probability 0.75), the reconstruction error of a mask is the mean squared
error between the reconstruction of the masked sample and the **original**
(normalized) sample, and the masks are drawn with a *stateless* RNG seeded
from ``(seed, sample_index, phase, repeat)``.

The statelessness is the property that makes the results independent of the
batch size (``--batch-size``) and of the chunking of the CSV file: the score of
a sample only depends on its global index, never on the batch it was
processed in.
"""

from __future__ import annotations

import numpy as np
import pytest

import new_data_eval.scoring as scoring
from new_data_eval.scoring import (
    MASK_PROBABILITY,
    PHASE_ORIGINAL,
    PHASE_VICTIM,
    load_model,
    reconstruction_errors,
    score_samples,
    validate_model,
)


class RecordingModel:
    """Identity model that records every (masked) input it receives.

    It makes the generated masks observable from the outside, which is how the
    tests below check the per-sample, stateless mask generation.
    """

    def __init__(self):
        self.batches: list[np.ndarray] = []

    def __call__(self, inputs, training=False):
        del training
        array = np.asarray(inputs, dtype=np.float32)
        self.batches.append(array.copy())
        return array

    @property
    def rows(self) -> np.ndarray:
        """Every recorded row, in call order."""
        if not self.batches:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(self.batches, axis=0)


def test_reconstruction_errors_shape_and_finite(tiny_model, meta):
    x = np.zeros((3, 77), dtype=np.float32)
    errors = reconstruction_errors(tiny_model, x, *meta, n_masks=10, batch_size=2)
    assert errors.shape == (3, 10)
    assert np.isfinite(errors).all()


def test_scores_are_batch_size_invariant(tiny_model, meta):
    x = np.random.default_rng(0).random((8, 77)).astype(np.float32)
    a = score_samples(tiny_model, x, *meta, mode="repo_plus", n_masks=20,
                      batch_size=2, seed=2019)
    b = score_samples(tiny_model, x, *meta, mode="repo_plus", n_masks=20,
                      batch_size=8, seed=2019)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_validate_model_rejects_wrong_input_width(tiny_model, meta):
    with pytest.raises(ValueError, match="77"):
        validate_model(tiny_model, *meta, benign_sample=np.zeros((2, 76)),
                       seed=2019)


def test_masks_are_drawn_per_sample_and_mask_features_with_probability_075(meta):
    model = RecordingModel()
    # Identical rows: any row difference can only come from the masks.
    x = np.full((2, 77), 0.5, dtype=np.float32)
    reconstruction_errors(model, x, *meta, n_masks=20, batch_size=8, seed=2019)

    rows = model.rows
    assert rows.shape == (40, 77)
    # The masks are drawn per sample: two identical samples do not share the
    # same mask set (a shared set would produce only 20 distinct rows).
    assert len(np.unique(rows, axis=0)) == 40
    # The rows are the masked samples: a masked feature is zero, a kept one
    # keeps the original value, so 25% of the features are kept.
    kept = float(np.count_nonzero(rows)) / rows.size
    assert 1.0 - MASK_PROBABILITY - 0.1 < kept < 1.0 - MASK_PROBABILITY + 0.1


def test_scoring_a_chunk_equals_scoring_the_whole_set(tiny_model, meta):
    x = np.random.default_rng(2).random((8, 77)).astype(np.float32)
    whole = score_samples(tiny_model, x, *meta, mode="repo", n_masks=20,
                          seed=2019)
    first = score_samples(tiny_model, x[:4], *meta, mode="repo", n_masks=20,
                          seed=2019, sample_indices=[0, 1, 2, 3])
    second = score_samples(tiny_model, x[4:], *meta, mode="repo", n_masks=20,
                           seed=2019, sample_indices=[4, 5, 6, 7])
    np.testing.assert_allclose(np.concatenate([first, second]), whole,
                               rtol=1e-6, atol=1e-6)


def test_sample_indices_shift_the_masks(meta):
    model = RecordingModel()
    x = np.random.default_rng(3).random((2, 77)).astype(np.float32)
    reconstruction_errors(model, x, *meta, n_masks=10, seed=2019,
                          sample_indices=[0, 1])
    first = model.rows.copy()
    model.batches.clear()
    reconstruction_errors(model, x, *meta, n_masks=10, seed=2019,
                          sample_indices=[10, 11])
    second = model.rows.copy()
    assert not np.array_equal(first, second)


def test_paired_seed_reuses_the_same_masks_across_phases(meta):
    model = RecordingModel()
    x = np.random.default_rng(4).random((2, 77)).astype(np.float32)

    reconstruction_errors(model, x, *meta, n_masks=10, seed=2019,
                          phase=PHASE_ORIGINAL)
    original = model.rows.copy()
    model.batches.clear()
    reconstruction_errors(model, x, *meta, n_masks=10, seed=2019,
                          phase=PHASE_VICTIM)
    victim = model.rows.copy()
    # Without the paired seed every phase uses its own masks.
    assert not np.array_equal(original, victim)

    model.batches.clear()
    reconstruction_errors(model, x, *meta, n_masks=10, seed=2019,
                          phase=PHASE_ORIGINAL, paired_seed=7)
    paired_original = model.rows.copy()
    model.batches.clear()
    reconstruction_errors(model, x, *meta, n_masks=10, seed=2019,
                          phase=PHASE_VICTIM, paired_seed=7)
    paired_victim = model.rows.copy()
    # With the paired seed the before/after phases share the masks.
    np.testing.assert_array_equal(paired_original, paired_victim)


def test_score_repeats_average_independent_scores(tiny_model, meta):
    x = np.random.default_rng(5).random((4, 77)).astype(np.float32)
    single = score_samples(tiny_model, x, *meta, mode="repo", n_masks=20,
                            seed=2019)
    repeated = score_samples(tiny_model, x, *meta, mode="repo", n_masks=20,
                             seed=2019, score_repeats=3)
    again = score_samples(tiny_model, x, *meta, mode="repo", n_masks=20,
                          seed=2019, score_repeats=3)

    assert repeated.shape == single.shape
    # The repeats use independent masks, so the averaged score differs ...
    assert not np.allclose(repeated, single)
    # ... but it is fully reproducible.
    np.testing.assert_allclose(again, repeated, rtol=1e-12, atol=1e-12)


def test_reconstruction_errors_average_the_repeats(monkeypatch, tiny_model, meta):
    # Two distinct, hand-written per-repeat error matrices.  Their mean is
    # different from either repeat, so a regression to errors[0] (or to any
    # single repeat) is caught.
    repeat_errors = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        ],
        dtype=np.float64,
    )

    def fake_per_repeat_errors(*args, **kwargs):
        assert kwargs["score_repeats"] == 2
        return repeat_errors.copy()

    monkeypatch.setattr(
        scoring, "_reconstruction_errors_normalized", fake_per_repeat_errors
    )

    errors = reconstruction_errors(
        tiny_model,
        np.zeros((2, 77), dtype=np.float32),
        *meta,
        n_masks=3,
        score_repeats=2,
    )

    np.testing.assert_allclose(errors, repeat_errors.mean(axis=0))
    assert not np.allclose(errors, repeat_errors[0])
    assert not np.allclose(errors, repeat_errors[1])


def test_reconstruction_errors_rejects_duplicate_sample_indices(tiny_model, meta):
    with pytest.raises(ValueError, match="duplicates"):
        reconstruction_errors(
            tiny_model,
            np.zeros((2, 77), dtype=np.float32),
            *meta,
            n_masks=5,
            sample_indices=[0, 0],
        )


def test_reconstruction_errors_rejects_wrong_feature_width(tiny_model, meta):
    with pytest.raises(ValueError, match="77"):
        reconstruction_errors(tiny_model, np.zeros((2, 76), dtype=np.float32),
                              *meta)


def test_reconstruction_errors_rejects_unknown_phase(tiny_model, meta):
    with pytest.raises(ValueError, match="phase"):
        reconstruction_errors(tiny_model, np.zeros((2, 77), dtype=np.float32),
                              *meta, phase="not-a-phase")


def test_reconstruction_errors_rejects_mismatched_sample_indices(tiny_model, meta):
    with pytest.raises(ValueError, match="sample_indices"):
        reconstruction_errors(tiny_model, np.zeros((2, 77), dtype=np.float32),
                              *meta, sample_indices=[0])


def test_score_samples_rejects_a_mask_count_not_divisible_by_five(meta):
    class ExplodingModel:
        def __call__(self, inputs, training=False):
            raise AssertionError("the model must not be called")

    x = np.zeros((2, 77), dtype=np.float32)
    with pytest.raises(ValueError, match="not divisible"):
        score_samples(ExplodingModel(), x, *meta, mode="repo_plus", n_masks=7)


def test_score_samples_rejects_an_unknown_mode(tiny_model, meta):
    x = np.zeros((2, 77), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown score mode"):
        score_samples(tiny_model, x, *meta, mode="not-a-mode")


def test_validate_model_accepts_a_sane_model(tiny_model, meta):
    report = validate_model(tiny_model, *meta,
                            benign_sample=np.full((2, 77), 5.0), seed=2019)
    assert report["input_width"] == 77
    assert report["benign_error"] < report["noise_error"]
    assert np.isfinite(report["benign_error"])
    assert np.isfinite(report["noise_error"])


def test_validate_model_rejects_a_model_that_does_not_reconstruct(tiny_model, meta):
    # A benign sample at the top of the normalized range is reconstructed
    # worse than uniform noise by the identity model.
    with pytest.raises(ValueError, match="noise"):
        validate_model(tiny_model, *meta, benign_sample=np.full((2, 77), 10.0),
                       seed=2019)


def test_validate_model_rejects_a_wrong_output_width(meta):
    class NarrowModel:
        def __call__(self, inputs, training=False):
            del training
            return np.asarray(inputs)[:, :76]

    with pytest.raises(ValueError, match="77"):
        validate_model(NarrowModel(), *meta, benign_sample=np.zeros((2, 77)))


def test_validate_model_rejects_non_finite_outputs(meta):
    class NaNModel:
        def __call__(self, inputs, training=False):
            del training
            return np.full(np.asarray(inputs).shape, np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        validate_model(NaNModel(), *meta, benign_sample=np.zeros((2, 77)))


def test_validate_model_rejects_a_fixed_batch_dimension(meta):
    class FixedBatchModel:
        def __call__(self, inputs, training=False):
            del training
            array = np.asarray(inputs)
            if array.shape[0] != 3:
                raise ValueError("expected a batch of three rows")
            return array

    with pytest.raises(ValueError, match="None"):
        validate_model(FixedBatchModel(), *meta,
                       benign_sample=np.zeros((3, 77)))


def test_load_model_loads_an_exported_saved_model(synthetic_project):
    model = load_model(synthetic_project.model_dir)
    train_min = np.zeros(77, dtype=np.float32)
    train_max = np.full(77, 10.0, dtype=np.float32)
    errors = reconstruction_errors(model, np.zeros((2, 77), dtype=np.float32),
                                   train_min, train_max, n_masks=5, batch_size=4)
    assert errors.shape == (2, 5)
    assert np.isfinite(errors).all()


def test_load_model_rejects_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing_model")
