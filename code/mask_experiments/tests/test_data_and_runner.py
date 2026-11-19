"""Tests for the CICIDS flow sampler and the baseline payload builder.

No TensorFlow is imported: the sampler and the payload builder are pure NumPy
and pandas so they run on a login node or on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mask_experiments.data import load_flow_meta, normalize_label, sample_flow_test
from mask_experiments.runner import build_baseline_payload, class_rates
from new_data_eval.scoring import ScoringCost


def _frame(rows: int, labels, seed: int = 0) -> pd.DataFrame:
    """Build a frame with the 85 column CICIDS2017 flow layout."""
    rng = np.random.default_rng(seed)
    values = {
        "Flow ID": np.arange(rows),
        "Src IP": ["10.0.0.1"] * rows,
        "Src Port": np.arange(rows),
        "Dst IP": ["10.0.0.2"] * rows,
        "Dst Port": rng.integers(0, 65536, rows),
        "Protocol": np.full(rows, 6),
        "Timestamp": ["2017-07-04 00:00:00"] * rows,
        "Flow Duration": rng.integers(0, 1000, rows),
    }
    for index in range(76):
        values[f"f{index}"] = rng.random(rows)
    values["Label"] = labels
    return pd.DataFrame(values)


def _write_days(root: Path, days=("Tuesday",)) -> None:
    directory = root / "flow_based"
    directory.mkdir(parents=True, exist_ok=True)
    for day in days:
        labels = ["BENIGN"] * 20 + ["DDoS"] * 10 + ["PortScan"] * 10
        _frame(40, labels, seed=len(day)).to_csv(
            directory / f"{day}-WH-generate-labeled.csv", index=False
        )


def test_sample_flow_test_is_capped_balanced_and_deterministic(tmp_path):
    _write_days(tmp_path)
    first = sample_flow_test(
        tmp_path, max_benign=8, max_per_attack=4, seed=2019, chunksize=7,
        days=("Tuesday",),
    )
    second = sample_flow_test(
        tmp_path, max_benign=8, max_per_attack=4, seed=2019, chunksize=7,
        days=("Tuesday",),
    )
    x, labels, indices = first

    np.testing.assert_array_equal(x, second[0])
    np.testing.assert_array_equal(labels, second[1])
    np.testing.assert_array_equal(indices, second[2])

    assert x.shape == (8 + 4 + 4, 77)
    assert x.dtype == np.float32
    counts = {label: int((labels == label).sum()) for label in set(labels)}
    assert counts == {"BENIGN": 8, "DDoS": 4, "PortScan": 4}
    assert len(np.unique(indices)) == len(indices)


def test_sample_flow_test_uses_all_rows_when_the_cap_is_larger(tmp_path):
    _write_days(tmp_path)
    x, labels, _ = sample_flow_test(
        tmp_path, max_benign=100, max_per_attack=100, days=("Tuesday",)
    )
    assert x.shape[0] == 40
    assert int((labels == "DDoS").sum()) == 10


def test_sample_flow_test_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_flow_data"):
        sample_flow_test(tmp_path, max_benign=1, max_per_attack=1)


def test_sample_flow_test_drops_invalid_rows(tmp_path):
    directory = tmp_path / "flow_based"
    directory.mkdir()
    frame = _frame(6, ["BENIGN"] * 6)
    frame.loc[1, "f0"] = np.inf
    frame.loc[3, "f1"] = np.nan
    frame.to_csv(directory / "Tuesday-WH-generate-labeled.csv", index=False)

    x, labels, indices = sample_flow_test(
        tmp_path, max_benign=10, max_per_attack=10, days=("Tuesday",)
    )
    assert x.shape[0] == 4
    assert np.isfinite(x).all()
    assert sorted(indices.tolist()) == [0, 2, 4, 5]


def test_normalize_label_maps_junk_and_web_attacks():
    assert normalize_label("BENIGN") == "BENIGN"
    assert normalize_label(" benign ") == "BENIGN"
    assert normalize_label("DDoS") == "DDoS"
    assert normalize_label("Web Attack - XSS") == "Web Attack"
    assert normalize_label("Web Attack - Sql Injection") == "Web Attack"
    assert normalize_label("Label") is None
    assert normalize_label("") is None
    assert normalize_label(np.nan) is None


def test_load_flow_meta_reads_the_77_feature_arrays(tmp_path):
    directory = tmp_path / "flow_based" / "x_train_meta"
    directory.mkdir(parents=True)
    np.save(directory / "train_min.npy", np.zeros(77, dtype=np.float32))
    np.save(directory / "train_max.npy", np.ones(77, dtype=np.float32))

    train_min, train_max = load_flow_meta(tmp_path)
    assert train_min.shape == (77,)
    np.testing.assert_array_equal(train_min, np.zeros(77))


def test_class_rates_and_baseline_payload_are_consistent(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights").write_bytes(b"abc")

    labels = np.array(["BENIGN"] * 10 + ["DDoS"] * 10)
    scores = np.concatenate([np.linspace(0.0, 0.9, 10), np.linspace(0.2, 1.0, 10)])
    cost = ScoringCost()
    cost.add_call(20, 0.5)

    payload = build_baseline_payload(
        scores={"repo": scores, "repo_plus": scores},
        labels=labels,
        x_raw=np.zeros((20, 77), dtype=np.float32),
        sample_indices=np.arange(20),
        dataset={"name": "cicids2017-flow", "days": ["Tuesday"]},
        model_dir=model_dir,
        n_masks=100,
        policy_name="iid",
        seed=2019,
        score_repeats=1,
        batch_size=100,
        cost=cost,
        runtime_seconds=1.5,
    )

    assert payload["schema_version"] == "mask_exp_v1"
    assert payload["dataset"]["n_samples"] == 20
    assert payload["dataset"]["n_features"] == 77
    assert payload["cost"]["masked_rows"] == 20
    assert payload["modes"]["repo"]["fpr"]["fpr_0.01"] == 0.1
    rates = payload["modes"]["repo"]["class_rates"]["fpr_0.01"]
    assert set(rates) == {"BENIGN", "DDoS"}
    assert rates["BENIGN"]["fpr"] == rates["BENIGN"]["rate"] == 0.1
    assert rates["DDoS"]["n"] == 10
    assert 0.0 <= payload["modes"]["repo"]["average_tpr"]["fpr_0.01"]["macro"] <= 1.0
    assert payload["config_hash"] == build_baseline_payload(
        scores={"repo": scores, "repo_plus": scores},
        labels=labels,
        x_raw=np.zeros((20, 77), dtype=np.float32),
        sample_indices=np.arange(20),
        dataset={"name": "cicids2017-flow", "days": ["Tuesday"]},
        model_dir=model_dir,
        n_masks=100,
        policy_name="iid",
        seed=2019,
        score_repeats=1,
        batch_size=100,
        cost=cost,
        runtime_seconds=1.5,
    )["config_hash"]


def test_class_rates_handles_absent_classes():
    scores = np.array([0.1, 0.9])
    labels = np.array(["BENIGN", "BENIGN"])
    rates = class_rates(scores, labels, 0.5, ["BENIGN", "DDoS"])
    assert rates["DDoS"] == {"n": 0, "rate": 0.0}
    assert rates["BENIGN"]["n"] == 2
