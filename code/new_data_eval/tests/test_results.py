"""Tests for the result schema, atomic JSON writing and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import new_data_eval.results as results_module
from new_data_eval.results import (
    MODEL_TYPE,
    SCHEMA_VERSION,
    build_adversarial_result,
    build_normal_result,
    config_hash,
    environment_manifest,
    file_sha256,
    git_commit,
    write_json_atomic,
)


def _normal_inputs():
    return {
        "score_mode": "repo_plus",
        "thresholds": {"fpr_0.01": 0.8, "fpr_0.10": 0.5},
        "scores": np.array([0.9, 0.7, 0.9, 0.4, 0.2]),
        "labels": np.array(["BENIGN", "BENIGN", "ddos", "ddos", "xss"]),
        "dataset": {
            "name": "new_data",
            "test_csv": "data/new_data/new_test.csv",
            "test_csv_sha256": "a" * 64,
            "n_samples": 5,
            "n_dropped_invalid": 0,
        },
        "scoring": {"n_masks": 100, "score_repeats": 1, "paired_scoring": False},
        "config": {
            "seed": 2019,
            "thresholds": {"fpr_0.01": 0.8, "fpr_0.10": 0.5},
            "sampling": {"mode": "random", "max_samples": None},
            "model_sha256": "b" * 64,
            "data_sha256": "a" * 64,
            "code_version": "0.1.0",
        },
        "runtime_seconds": 1.25,
        "started_at": "2026-11-09T10:00:00+00:00",
        "finished_at": "2026-11-09T10:00:02+00:00",
        "environment": {"python": "3.13.12", "numpy": "2.2.6"},
        "git_commit": "c" * 40,
    }


def _adversarial_inputs():
    return {
        "attack_type": "ddos",
        "variant": "primary",
        "score_mode": "repo_plus",
        "dataset": {
            "n_available": 10,
            "n_samples": 4,
            "sample_mode": "random",
            "seed": 2019,
        },
        "thresholds": {"fpr_0.10": 0.5, "attack": 0.5, "policy": "calibrated"},
        "scoring": {
            "score_mode": "repo_plus",
            "n_masks": 100,
            "score_repeats": 1,
            "paired_scoring": False,
        },
        "benign_scores": np.array([0.1, 0.9]),
        "normal_scores": np.array([0.9, 0.9, 0.1, 0.1]),
        "victim_scores": np.array([0.1, 0.9, 0.1, 0.1]),
        "counts": {
            "n": 4,
            "fooled": 2,
            "evaded": 1,
            "backup": 1,
            "cons_as_mal": 1,
            "cons_as_ben": 1,
        },
        "config": {
            "seed": 2019,
            "thresholds": {"fpr_0.10": 0.5, "attack": 0.5},
            "sampling": {"mode": "random", "max_samples": 4},
            "model_sha256": "b" * 64,
            "data_sha256": "a" * 64,
            "code_version": "0.1.0",
        },
        "runtime_seconds": 0.5,
        "started_at": "2026-11-09T10:00:00+00:00",
        "finished_at": "2026-11-09T10:00:01+00:00",
        "environment": {"python": "3.13.12", "numpy": "2.2.6"},
        "git_commit": "c" * 40,
    }


def test_write_json_atomic_replaces_file(tmp_path):
    path = tmp_path / "result.json"
    write_json_atomic(path, {"a": 1})
    write_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}


def test_write_json_atomic_creates_parent_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "nested" / "result.json"
    write_json_atomic(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [item.name for item in path.parent.iterdir()] == ["result.json"]


def test_write_json_atomic_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        results_module, "_fsync_directory", lambda path: calls.append(Path(path))
    )
    path = tmp_path / "nested" / "result.json"
    write_json_atomic(path, {"a": 1})
    assert calls == [path.parent]


def test_file_sha256_matches_the_known_digest(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_config_hash_is_stable_for_equal_configs():
    config = _normal_inputs()["config"]
    reordered = dict(reversed(list(config.items())))
    assert config_hash(config) == config_hash(reordered)
    assert len(config_hash(config)) == 64


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("seed", None, 2020),
        ("thresholds", "fpr_0.01", 0.9),
        ("sampling", "mode", "first"),
        ("model_sha256", None, "d" * 64),
        ("data_sha256", None, "e" * 64),
    ],
)
def test_config_hash_changes_with_relevant_config_sections(section, key, value):
    base = _normal_inputs()["config"]
    changed = json.loads(json.dumps(base))
    if key is None:
        changed[section] = value
    else:
        changed[section][key] = value
    assert config_hash(base) != config_hash(changed)


def test_environment_manifest_reports_versions():
    manifest = environment_manifest()
    assert manifest["python"]
    assert manifest["numpy"]
    assert manifest["pandas"]
    assert "platform" in manifest


def test_git_commit_is_hex_or_none():
    commit = git_commit()
    assert commit is None or (
        isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    )


def test_build_normal_result_matches_the_schema():
    payload = build_normal_result(**_normal_inputs())

    assert payload["schema_version"] == SCHEMA_VERSION == "new_data_v1"
    assert payload["status"] == "completed"
    assert payload["experiment"] == "normal"
    assert payload["model_type"] == MODEL_TYPE == "new_data_flow"
    assert payload["score_mode"] == "repo_plus"
    assert payload["variant"] == "primary"

    assert payload["thresholds"] == {"fpr_0.01": 0.8, "fpr_0.10": 0.5}
    assert set(payload["fpr"]) == {
        "at_fpr_0.01_threshold",
        "at_fpr_0.10_threshold",
    }
    assert set(payload["class_rates"]) == {"at_fpr_0.01", "at_fpr_0.10"}
    assert set(payload["average_tpr"]) == {"at_fpr_0.01", "at_fpr_0.10"}

    for threshold_name in ("at_fpr_0.01", "at_fpr_0.10"):
        rates = payload["class_rates"][threshold_name]
        assert rates["BENIGN"]["n"] == 2
        assert "fpr" in rates["BENIGN"]
        for attack in ("ddos", "xss"):
            assert set(rates[attack]) == {"n", "tpr", "low_n"}
        assert set(payload["average_tpr"][threshold_name]) == {
            "macro",
            "weighted",
            "macro_excluding_low_n",
        }

    assert payload["dataset"]["test_csv_sha256"] == "a" * 64
    assert payload["dataset"]["n_samples"] == 5
    assert payload["dataset"]["n_dropped_invalid"] == 0
    assert payload["scoring"] == {
        "n_masks": 100,
        "score_repeats": 1,
        "paired_scoring": False,
    }
    assert payload["runtime_seconds"] == 1.25
    assert payload["config"] == _normal_inputs()["config"]
    assert payload["config_hash"] == build_normal_result(
        **_normal_inputs()
    )["config_hash"]
    assert payload["environment"] == {"python": "3.13.12", "numpy": "2.2.6"}
    assert payload["git_commit"] == "c" * 40
    assert payload["started_at"] == "2026-11-09T10:00:00+00:00"
    assert payload["finished_at"] == "2026-11-09T10:00:02+00:00"

    # The payload itself must be JSON serializable (no NumPy scalars leak).
    assert json.loads(json.dumps(payload))["class_rates"]["at_fpr_0.01"]["ddos"][
        "tpr"
    ] == 0.5


def test_build_normal_result_computes_fpr_and_average_tpr():
    payload = build_normal_result(**_normal_inputs())
    # BENIGN scores: 0.9 and 0.7; the 0.8 threshold flags one.
    assert payload["fpr"]["at_fpr_0.01_threshold"] == pytest.approx(0.5)
    # The 0.5 threshold flags both benign rows.
    assert payload["fpr"]["at_fpr_0.10_threshold"] == pytest.approx(1.0)
    # Attacks at the 0.5 threshold: ddos 1/2, xss 0/1 and the seven absent
    # classes 0/0; macro is over all nine attacks, weighted by n.
    assert payload["average_tpr"]["at_fpr_0.10"]["macro"] == pytest.approx(0.5 / 9)
    assert payload["average_tpr"]["at_fpr_0.10"]["weighted"] == pytest.approx(1 / 3)


def test_build_normal_result_defaults_variant_and_status():
    inputs = _normal_inputs()
    inputs.pop("git_commit")
    payload = build_normal_result(**inputs)
    assert payload["variant"] == "primary"
    assert payload["status"] == "completed"
    assert payload["git_commit"] is None or isinstance(payload["git_commit"], str)
    assert payload["config_hash"] == build_normal_result(**inputs)["config_hash"]


def test_build_normal_result_rejects_a_missing_required_key():
    inputs = _normal_inputs()
    inputs.pop("scores")
    with pytest.raises(TypeError, match="scores"):
        build_normal_result(**inputs)


def test_build_normal_result_rejects_dataset_score_count_mismatch():
    inputs = _normal_inputs()
    inputs["dataset"]["n_samples"] = 4
    with pytest.raises(ValueError, match="n_samples"):
        build_normal_result(**inputs)


def test_build_normal_result_config_hash_covers_run_parameters():
    base = build_normal_result(**_normal_inputs())["config_hash"]

    # The config mapping is unchanged: only the separately passed run
    # parameters change, and each must still alter the hash.
    changed = _normal_inputs()
    changed["thresholds"] = {"fpr_0.01": 0.81, "fpr_0.10": 0.5}
    assert build_normal_result(**changed)["config_hash"] != base

    changed = _normal_inputs()
    changed["scoring"] = {
        "n_masks": 50,
        "score_repeats": 2,
        "paired_scoring": True,
    }
    assert build_normal_result(**changed)["config_hash"] != base

    changed = _normal_inputs()
    changed["dataset"]["test_csv_sha256"] = "f" * 64
    assert build_normal_result(**changed)["config_hash"] != base

    # The variant and the code revision are part of the run identity: a smoke
    # result must never share a hash with the primary one, and a rerun on a
    # different commit must not either.
    changed = _normal_inputs()
    changed["variant"] = "smoke"
    assert build_normal_result(**changed)["config_hash"] != base

    changed = _normal_inputs()
    changed["git_commit"] = "d" * 40
    assert build_normal_result(**changed)["config_hash"] != base


def test_build_adversarial_result_matches_the_schema():
    payload = build_adversarial_result(**_adversarial_inputs())

    assert payload["schema_version"] == "new_data_v1"
    assert payload["status"] == "completed"
    assert payload["experiment"] == "adversarial"
    assert payload["model_type"] == "new_data_flow"
    assert payload["score_mode"] == "repo_plus"
    assert payload["attack_type"] == "ddos"
    assert payload["variant"] == "primary"

    assert payload["dataset"] == {
        "n_available": 10,
        "n_samples": 4,
        "sample_mode": "random",
        "seed": 2019,
    }
    assert payload["thresholds"] == {
        "fpr_0.10": 0.5,
        "attack": 0.5,
        "policy": "calibrated",
    }
    assert payload["scoring"] == {
        "score_mode": "repo_plus",
        "n_masks": 100,
        "score_repeats": 1,
        "paired_scoring": False,
    }
    assert payload["fpr_at_attack_threshold"] == pytest.approx(0.5)
    assert payload["normal_tpr_at_attack_threshold"] == pytest.approx(0.5)
    assert payload["victim_tpr"] == pytest.approx(0.25)
    assert payload["attack_success_rate"] == pytest.approx(0.75)
    assert payload["conditional_drop"] == pytest.approx(0.5)
    assert payload["fooled_rate"] == pytest.approx(0.5)
    assert payload["evaded_rate"] == pytest.approx(0.25)
    assert payload["backup_rate"] == pytest.approx(0.25)
    assert payload["counts"] == _adversarial_inputs()["counts"]
    assert payload["runtime_seconds"] == 0.5
    assert payload["config_hash"] == build_adversarial_result(
        **_adversarial_inputs()
    )["config_hash"]
    assert payload["git_commit"] == "c" * 40


def test_build_adversarial_result_accepts_precomputed_rates():
    inputs = _adversarial_inputs()
    for key in ("benign_scores", "normal_scores", "victim_scores"):
        inputs.pop(key)
    inputs.update(
        {
            "fpr_at_attack_threshold": 0.2,
            "normal_tpr_at_attack_threshold": 0.0,
            "victim_tpr": 0.1,
        }
    )
    payload = build_adversarial_result(**inputs)
    assert payload["fpr_at_attack_threshold"] == 0.2
    assert payload["victim_tpr"] == 0.1
    assert payload["conditional_drop"] is None


def test_build_adversarial_result_rejects_inconsistent_score_counts():
    inputs = _adversarial_inputs()
    inputs["victim_scores"] = np.array([0.1])
    with pytest.raises(ValueError, match="n"):
        build_adversarial_result(**inputs)


def test_build_adversarial_result_rejects_dataset_count_mismatch():
    inputs = _adversarial_inputs()
    inputs["dataset"]["n_samples"] = 5
    with pytest.raises(ValueError, match="n_samples"):
        build_adversarial_result(**inputs)


def test_build_adversarial_result_config_hash_covers_run_parameters():
    base = build_adversarial_result(**_adversarial_inputs())["config_hash"]

    # The config mapping is unchanged in every case below.
    changed = _adversarial_inputs()
    changed["thresholds"] = {
        "fpr_0.10": 0.5,
        "attack": 0.6,
        "policy": "calibrated",
    }
    assert build_adversarial_result(**changed)["config_hash"] != base

    changed = _adversarial_inputs()
    changed["scoring"] = {
        "score_mode": "repo_plus",
        "n_masks": 50,
        "score_repeats": 2,
        "paired_scoring": True,
    }
    assert build_adversarial_result(**changed)["config_hash"] != base

    changed = _adversarial_inputs()
    changed["dataset"]["seed"] = 2020
    assert build_adversarial_result(**changed)["config_hash"] != base

    # The variant and the code revision are part of the run identity.
    changed = _adversarial_inputs()
    changed["variant"] = "sensitivity_first_n"
    assert build_adversarial_result(**changed)["config_hash"] != base

    changed = _adversarial_inputs()
    changed["git_commit"] = "d" * 40
    assert build_adversarial_result(**changed)["config_hash"] != base

    # Outcome counts are not part of the identity: a different outcome with
    # the same dataset sampling and configuration must keep the same hash.
    changed = _adversarial_inputs()
    changed["counts"] = {
        "n": 4,
        "fooled": 1,
        "evaded": 1,
        "backup": 0,
        "cons_as_mal": 2,
        "cons_as_ben": 1,
    }
    assert build_adversarial_result(**changed)["config_hash"] == base


@pytest.mark.parametrize(
    "counts, match",
    [
        (
            {
                "n": 4,
                "fooled": 3,
                "evaded": 1,
                "backup": 1,
                "cons_as_mal": 1,
                "cons_as_ben": 0,
            },
            "fooled",
        ),
        (
            {
                "n": 1,
                "fooled": 2,
                "evaded": 1,
                "backup": 1,
                "cons_as_mal": 0,
                "cons_as_ben": 0,
            },
            "fooled",
        ),
        (
            {
                "n": 4,
                "fooled": 2,
                "evaded": 1,
                "backup": 1,
                "cons_as_mal": 0,
                "cons_as_ben": 1,
            },
            "cons_as_mal",
        ),
        (
            {
                "n": -1,
                "fooled": 0,
                "evaded": 0,
                "backup": 0,
                "cons_as_mal": 0,
                "cons_as_ben": 0,
            },
            "negative",
        ),
    ],
)
def test_build_adversarial_result_rejects_invalid_counts(counts, match):
    inputs = _adversarial_inputs()
    inputs["counts"] = counts
    with pytest.raises(ValueError, match=match):
        build_adversarial_result(**inputs)


def test_results_are_written_as_one_atomic_json_file(tmp_path):
    payload = build_normal_result(**_normal_inputs())
    path = tmp_path / "normal" / "repo_plus.json"
    write_json_atomic(path, payload)
    assert json.loads(path.read_text())["model_type"] == MODEL_TYPE
    assert [item.name for item in path.parent.iterdir()] == ["repo_plus.json"]
