"""Tests for the per-attack merge step used by the SLURM arrays."""

from __future__ import annotations

import json

import pytest

from mask_experiments.cascade_attack import _decide
from mask_experiments.merge import (
    merge_adversarial,
    merge_cascade_attack,
    merge_distill_attack,
    merge_variance,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _adversarial(attack: str, n: int = 2) -> dict:
    return {
        "dataset": {"name": "cicids2017-flow", "attacks": {attack: n}},
        "variants": {
            "iid-25": {
                "policy": "iid",
                "n_masks": 25,
                "real_threshold": 0.03,
                "attack_threshold": 0.02,
                "attacks": {
                    attack: {
                        "n": n,
                        "normal_tpr": 0.5,
                        "victim_tpr": 0.25,
                        "fooled": 1,
                        "kinds": {"no_change": 0},
                        "outcomes": [{"masks_evaluated": 1000}] * n,
                    }
                },
                "cost": {"masked_rows": 10, "model_calls": 1, "model_seconds": 0.1},
            }
        },
    }


def test_merge_adversarial_combines_attacks_and_costs(tmp_path):
    _write(tmp_path / "adversarial" / "a.json", _adversarial("A"))
    _write(tmp_path / "adversarial" / "b.json", _adversarial("B", n=3))
    out = tmp_path / "adversarial_all.json"
    merged = merge_adversarial(tmp_path / "adversarial", out)

    assert set(merged["variants"]["iid-25"]["attacks"]) == {"A", "B"}
    assert merged["dataset"]["attacks"] == {"A": 2, "B": 3}
    assert merged["variants"]["iid-25"]["cost"]["masked_rows"] == 20
    assert json.loads(out.read_text())["variants"]["iid-25"]["n_masks"] == 25


def test_merge_cascade_attack_combines_attacks(tmp_path):
    base = {
        "configs": {"cheap5": {"n_cheap": 5}},
        "attacks": {},
        "dataset": {"attacks": {}},
    }
    for attack in ("A", "B"):
        payload = json.loads(json.dumps(base))
        payload["attacks"] = {
            attack: {"cheap5": {"n": 2, "victim_tpr": 0.5}}
        }
        _write(tmp_path / "cascade_attack" / f"{attack}.json", payload)
    merged = merge_cascade_attack(
        tmp_path / "cascade_attack", tmp_path / "cascade_attack_all.json"
    )
    assert set(merged["attacks"]) == {"A", "B"}
    assert merged["configs"]["cheap5"]["n_cheap"] == 5
    assert merged["dataset"]["attacks"] == {"A": 2, "B": 2}


def test_merge_distill_attack_combines_attacks(tmp_path):
    base = {"thresholds": {}, "attacks": {}, "dataset": {"attacks": {}}}
    for attack in ("A", "B"):
        payload = json.loads(json.dumps(base))
        payload["attacks"] = {attack: {"n": 4, "student_victim_tpr": 0.0}}
        _write(tmp_path / "distill_attack" / f"{attack}.json", payload)
    merged = merge_distill_attack(
        tmp_path / "distill_attack", tmp_path / "distill_attack_all.json"
    )
    assert set(merged["attacks"]) == {"A", "B"}
    assert merged["dataset"]["attacks"] == {"A": 4, "B": 4}


def _variance(policy: str) -> dict:
    return {
        "schema_version": "mask_exp_v1",
        "experiment": "variance",
        "dataset": {"name": "cicids2017-flow", "n_samples": 10},
        "scoring": {
            "n_masks": [5, 25],
            "policies": [policy],
            "score_repeats": 2,
        },
        "results": {policy: {"5": {"cost": {"samples": 10}}}},
        "runtime_seconds": 1.5,
    }


def test_merge_variance_combines_policy_partials(tmp_path):
    for policy in ("iid", "balanced", "covering"):
        _write(tmp_path / f"variance_flow_{policy}.json", _variance(policy))

    merged = merge_variance(tmp_path)

    assert set(merged["results"]) == {"iid", "balanced", "covering"}
    assert sorted(merged["scoring"]["policies"]) == ["balanced", "covering", "iid"]
    assert merged["runtime_seconds"] == 4.5
    written = json.loads((tmp_path / "variance_flow.json").read_text())
    assert set(written["results"]) == {"iid", "balanced", "covering"}


def test_merge_variance_rejects_incompatible_partials(tmp_path):
    _write(tmp_path / "variance_flow_iid.json", _variance("iid"))
    other = _variance("balanced")
    other["dataset"]["n_samples"] = 99
    _write(tmp_path / "variance_flow_balanced.json", other)

    with pytest.raises(ValueError):
        merge_variance(tmp_path)


def test_merge_variance_without_partials_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_variance(tmp_path)


def test_cascade_decision_requires_both_stages():
    import numpy as np

    cheap = np.array([0.0, 1.0, 1.0])
    full = np.array([0.0, 1.0, 0.0])
    decision = _decide(cheap, full, t_low=0.5, threshold=0.5)
    # first is bypassed, second is detected by the full ensemble, third is
    # routed but below the full threshold.
    assert decision.tolist() == [False, True, False]
