"""Regression tests for the on-disk sample cache used by the experiments."""

from __future__ import annotations

import numpy as np

from mask_experiments import adversarial, cascade, distill, variance
from mask_experiments.variance import _load_npz


def test_load_npz_accepts_object_labels(tmp_path):
    """Older caches stored the string labels as object arrays."""
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        x=np.zeros((2, 3), dtype=np.float32),
        labels=np.array(["BENIGN", "DDoS"], dtype=object),
        indices=np.array([1, 2], dtype=np.int64),
    )

    x, labels, indices = _load_npz(path)

    assert x.shape == (2, 3)
    assert labels.tolist() == ["BENIGN", "DDoS"]
    assert indices.tolist() == [1, 2]


def test_mask_clis_do_not_default_to_a_machine_specific_cache():
    for module in (variance, cascade, distill, adversarial):
        args = module._build_parser().parse_args([])
        assert args.cache == "", module.__name__
