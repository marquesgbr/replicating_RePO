"""Tests for the per-attack merge step used by the SLURM arrays."""

from __future__ import annotations

import json

from mask_experiments.cascade_attack import _decide
from mask_experiments.merge import (
    merge_adversarial,
    merge_cascade_attack,
    merge_distill_attack,
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


def test_cascade_decision_requires_both_stages():
    import numpy as np

    cheap = np.array([0.0, 1.0, 1.0])
    full = np.array([0.0, 1.0, 0.0])
    decision = _decide(cheap, full, t_low=0.5, threshold=0.5)
    # first is bypassed, second is detected by the full ensemble, third is
    # routed but below the full threshold.
    assert decision.tolist() == [False, True, False]
