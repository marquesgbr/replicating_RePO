from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CLUSTER_DIR = Path(__file__).resolve().parents[1]
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import metrics  # noqa: E402


def test_threshold_at_fpr_flow_ceil():
    scores = np.arange(1, 101, dtype=float)
    assert metrics.threshold_at_fpr(scores, 0.01, method="ceil") == 100.0
    assert metrics.threshold_at_fpr(scores, 0.10, method="ceil") == 91.0


def test_threshold_at_fpr_packet_round():
    scores = np.arange(1, 101, dtype=float)
    assert metrics.threshold_at_fpr(scores, 0.01, method="round") == 100.0
    assert metrics.threshold_at_fpr(scores, 0.10, method="round") == 91.0


def test_threshold_rejects_empty_and_invalid_fpr():
    with pytest.raises(ValueError):
        metrics.threshold_at_fpr([], 0.01)
    with pytest.raises(ValueError):
        metrics.threshold_at_fpr([1, 2, 3], 0.0)
    with pytest.raises(ValueError):
        metrics.threshold_at_fpr([1, 2, 3], 0.01, method="unknown")


def test_aggregate_mask_errors_repo_and_plus():
    errors = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )
    repo = metrics.aggregate_mask_errors(errors, mode="repo")
    plus = metrics.aggregate_mask_errors(errors, mode="repo_plus", groups=5)
    np.testing.assert_allclose(repo, [3.0, 2.0])
    np.testing.assert_allclose(plus, [15.0, 10.0])


def test_aggregate_mask_errors_requires_divisible_groups():
    errors = np.ones((2, 7))
    with pytest.raises(ValueError):
        metrics.aggregate_mask_errors(errors, mode="repo_plus", groups=5)


def test_compute_class_rates_with_string_labels():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    labels = np.array(["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk", "DDoS"])
    rates = metrics.compute_class_rates(
        scores, labels, ("BENIGN", "DoS Hulk", "DDoS"), threshold=0.5
    )
    assert rates["BENIGN"]["rate"] == 0.5
    assert rates["DoS Hulk"]["rate"] == 0.5
    assert rates["DDoS"]["rate"] == 1.0
    assert rates["DDoS"]["count"] == 1


def test_compute_class_rates_with_integer_labels():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    labels = np.array([0, 0, 1, 1, 3])
    rates = metrics.compute_class_rates(
        scores, labels, ("Benign", "FTP-Patator", "SSH-Patator", "DDoS"), threshold=0.5
    )
    assert rates["Benign"]["rate"] == 0.5
    assert rates["FTP-Patator"]["rate"] == 0.5
    assert rates["SSH-Patator"] == {"rate": None, "count": 0}
    assert rates["DDoS"]["rate"] == 1.0


def test_average_attack_tpr_excludes_benign_and_web():
    rates = {
        "BENIGN": {"rate": 0.01},
        "Web Attack": {"rate": 0.99},
        "A": {"rate": 0.2},
        "B": {"rate": 0.4},
    }
    assert metrics.average_attack_tpr(rates) == pytest.approx(0.3)


def test_config_hash_is_stable_and_sensitive():
    first = metrics.config_hash({"a": 1, "b": [2, 3]})
    second = metrics.config_hash({"b": [2, 3], "a": 1})
    third = metrics.config_hash({"a": 2, "b": [2, 3]})
    assert first == second
    assert first != third


def test_sanitize_name():
    assert metrics.sanitize_name("DoS Slowhttptest") == "DoS-Slowhttptest"
    assert metrics.sanitize_name("a/b c") == "a-b-c"
    assert metrics.sanitize_name("   ") == "item"


def test_expected_attack_lists_match_paper():
    assert len(metrics.FLOW_ATTACKS) == 11
    assert len(metrics.PACKET_ATTACKS) == 11
    assert "Web-Attack" not in metrics.PACKET_ATTACKS
    assert "Web Attack" not in metrics.FLOW_ATTACKS
    assert metrics.expected_attacks("flow") == metrics.FLOW_ATTACKS
    assert metrics.expected_attacks("packet") == metrics.PACKET_ATTACKS


def test_write_json_atomic_serializes_numpy(tmp_path):
    target = tmp_path / "nested" / "result.json"
    metrics.write_json_atomic(target, {"value": np.float32(1.5), "array": np.arange(3)})
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["value"] == pytest.approx(1.5)
    assert payload["array"] == [0, 1, 2]


def test_model_fingerprint_changes_with_content(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "saved_model.pb").write_bytes(b"first")
    first = metrics.model_fingerprint(model_dir)
    assert first["exists"] is True
    (model_dir / "saved_model.pb").write_bytes(b"second")
    second = metrics.model_fingerprint(model_dir)
    assert first["sha256"] != second["sha256"]
