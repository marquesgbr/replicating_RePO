"""Shared fixtures for the ``new_data_eval`` test suite.

The repository keeps every Python module inside ``code/``.  Because ``code``
is also the name of a standard library module, the package is imported as
``new_data_eval`` with ``<repo>/code`` on ``sys.path`` -- the same convention
already used by the notebooks (``import load_model_compat``).

All fixtures are function scoped on purpose: tests must never share result
directories, CSVs or models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS_DIR.parent
CODE_ROOT = PACKAGE_DIR.parent
REPO_ROOT = CODE_ROOT.parent

for _path in (str(REPO_ROOT), str(CODE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

N_FEATURES = 77
TINY_LABEL = "BENIGN"


def _synthetic_frame(
    labels, seed=2019, invalid=(), *, feature_scale=100.0, dst_port_max=65535
):
    """Build a DataFrame with the canonical 85 column layout.

    ``invalid`` is a sequence of ``(row_index, column, value)`` triples used
    to inject ``inf``/``nan`` values inside the feature range.
    ``feature_scale``/``dst_port_max`` let the training fixture stay inside the
    committed ``0..10`` normalization range, which is what ``validate_model``
    needs to see a sane reconstruction error.
    """
    from new_data_eval.data import EXPECTED_COLUMNS

    rng = np.random.default_rng(seed)
    n_rows = len(labels)
    columns = {}
    for name in EXPECTED_COLUMNS:
        if name == "column00":
            columns[name] = np.arange(n_rows, dtype=np.int64)
        elif name == "Flow ID":
            columns[name] = [f"10.0.0.1-10.0.0.2-{50000 + i}" for i in range(n_rows)]
        elif name == "Src IP":
            columns[name] = ["10.0.0.1"] * n_rows
        elif name == "Src Port":
            columns[name] = np.arange(1024, 1024 + n_rows, dtype=np.int64)
        elif name == "Dst IP":
            columns[name] = ["10.0.0.2"] * n_rows
        elif name == "Dst Port":
            columns[name] = rng.integers(1, dst_port_max, size=n_rows)
        elif name == "Protocol":
            columns[name] = np.full(n_rows, 6, dtype=np.int64)
        elif name == "Timestamp":
            columns[name] = np.arange(n_rows, dtype=np.float64) * 0.5
        elif name == "Label":
            columns[name] = list(labels)
        else:
            columns[name] = rng.random(n_rows) * feature_scale

    frame = pd.DataFrame(columns, columns=list(EXPECTED_COLUMNS))
    for row_index, column, value in invalid:
        frame.at[row_index, column] = value
    return frame


def _tiny_keras_model():
    """A deterministic 77 -> 77 model (identity reconstruction)."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(N_FEATURES,)),
            tf.keras.layers.Dense(N_FEATURES, activation="linear"),
        ]
    )
    model.layers[-1].set_weights(
        [np.eye(N_FEATURES, dtype=np.float32), np.zeros(N_FEATURES, dtype=np.float32)]
    )
    return model


@pytest.fixture
def tiny_model():
    """A tiny Keras model with 77 inputs (used by the scoring/attack tests)."""
    return _tiny_keras_model()


@pytest.fixture
def meta():
    """A pair of 77 element normalization arrays (``train_min``/``train_max``)."""
    train_min = np.zeros(N_FEATURES, dtype=np.float32)
    train_max = np.full(N_FEATURES, 10.0, dtype=np.float32)
    return train_min, train_max


@pytest.fixture
def synthetic_frame():
    """Factory building canonical frames (``labels``, ``seed``, ``invalid``)."""
    return _synthetic_frame


@pytest.fixture
def synthetic_project(tmp_path):
    """A self contained project with synthetic CSVs, metadata and a model.

    The generated test CSV contains ``BENIGN`` plus the nine attack labels and
    two rows with ``inf``/``nan`` inside the feature range, which must be
    dropped (and counted) by the validator.
    """
    from new_data_eval.data import ATTACK_LABELS, EXPECTED_COLUMNS

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    model_dir = tmp_path / "model"
    results_dir = tmp_path / "results"
    (results_dir / "normal").mkdir(parents=True)

    train_csv = data_dir / "new_train.csv"
    test_csv = data_dir / "new_test.csv"
    unknown_label_csv = data_dir / "unknown_label.csv"

    train_frame = _synthetic_frame(
        [TINY_LABEL] * 10, seed=11, feature_scale=1.0, dst_port_max=3
    )
    train_frame.to_csv(train_csv, index=False)

    labels = (
        [TINY_LABEL]
        + [label for label in ATTACK_LABELS for _ in range(3)]
        + [TINY_LABEL, TINY_LABEL]
    )
    feature_columns = [EXPECTED_COLUMNS[index] for index in range(8, 84)]
    invalid = [
        (len(labels) - 2, feature_columns[0], np.inf),
        (len(labels) - 1, feature_columns[5], np.nan),
    ]
    test_frame = _synthetic_frame(labels, seed=22, invalid=invalid)
    test_frame.to_csv(test_csv, index=False)

    unknown_frame = test_frame.copy()
    unknown_frame.loc[5, "Label"] = "not_an_attack"
    unknown_frame.to_csv(unknown_label_csv, index=False)

    np.save(meta_dir / "train_min.npy", np.zeros(N_FEATURES, dtype=np.float32))
    np.save(meta_dir / "train_max.npy", np.full(N_FEATURES, 10.0, dtype=np.float32))

    _tiny_keras_model().export(str(model_dir))

    from new_data_eval.cli import _sha256_path
    from new_data_eval.data import sha256_file

    normal_payload = {
        "schema_version": "new_data_v1",
        "status": "completed",
        "model_type": "new_data_flow",
        "score_mode": "repo_plus",
        "dataset": {
            "name": "new_data",
            "test_csv": str(test_csv),
            "test_csv_sha256": sha256_file(test_csv),
            "n_samples": len(labels),
            "n_dropped_invalid": 2,
        },
        "scoring": {
            "n_masks": 5,
            "score_repeats": 1,
            "paired_scoring": False,
        },
        "thresholds": {"fpr_0.01": 0.05, "fpr_0.10": 0.123},
        "fpr": {
            "at_fpr_0.01_threshold": 0.05,
            "at_fpr_0.10_threshold": 0.10,
        },
        "config": {
            "seed": 2019,
            "model_sha256": _sha256_path(model_dir),
            "meta_sha256": {
                "train_min": sha256_file(meta_dir / "train_min.npy"),
                "train_max": sha256_file(meta_dir / "train_max.npy"),
            },
            "sampling": {"mode": "full", "max_samples": None},
        },
        "config_hash": "a" * 64,
        "variant": "primary",
    }
    normal_repo_plus_json = results_dir / "normal" / "repo_plus.json"
    normal_repo_plus_json.write_text(
        json.dumps(normal_payload, indent=2), encoding="utf-8"
    )

    return SimpleNamespace(
        root=tmp_path,
        data_dir=data_dir,
        train_csv=train_csv,
        test_csv=test_csv,
        unknown_label_csv=unknown_label_csv,
        meta_dir=meta_dir,
        model_dir=model_dir,
        results_dir=results_dir,
        normal_repo_plus_json=normal_repo_plus_json,
    )
