"""Tests for the ``adversarial`` and ``smoke`` CLI entrypoints.

The brief of Task 7 defines six integration tests.  A few extra tests cover
the two integration details that are easiest to get silently wrong: the
checkpoint configuration hash (immutable parameters only) and the forwarding
of ``--score-repeats`` to the original/victim scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import new_data_eval.cli as cli
from new_data_eval.attack import AttackOutcome
from new_data_eval.cli import (
    checkpoint_path,
    load_normal_threshold,
    main,
    result_path,
)
from new_data_eval.data import ATTACK_LABELS
from new_data_eval.scoring import PHASE_ORIGINAL, PHASE_VICTIM


def test_adversarial_reads_threshold_from_normal_json(synthetic_project):
    threshold = load_normal_threshold(synthetic_project.results_dir)
    assert threshold == 0.123


def test_adversarial_rejects_unknown_label(synthetic_project):
    exit_code = main([
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "not_an_attack",
        "--n-masks", "5",
    ])
    assert exit_code != 0
    assert not list((synthetic_project.results_dir / "adversarial").rglob("*.json"))


def test_attack_label_is_case_insensitive(synthetic_project):
    exit_code = main([
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "DDoS",
        "--max-samples", "1",
        "--n-masks", "5",
        "--variant", "primary",
    ])
    assert exit_code == 0
    assert (synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json").exists()


def test_adversarial_writes_variant_directory(synthetic_project):
    exit_code = main([
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "ddos",
        "--max-samples", "2",
        "--n-masks", "5",
        "--variant", "sensitivity_first_n",
    ])
    assert exit_code == 0
    assert (synthetic_project.results_dir / "adversarial" / "sensitivity_first_n" / "ddos.json").exists()
    assert not (synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json").exists()


def test_checkpoint_resume_does_not_duplicate_samples(synthetic_project):
    args = [
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "ddos",
        "--max-samples", "2",
        "--n-masks", "5",
        "--checkpoint-every", "1",
    ]
    assert main(args) == 0
    result_path = synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
    first = json.loads(result_path.read_text())
    assert first["counts"]["n"] == 2

    result_path.unlink()
    assert main(args) == 0
    second = json.loads(result_path.read_text())
    assert second["counts"]["n"] == 2
    assert second["counts"] == first["counts"]


def test_smoke_is_isolated_from_primary(synthetic_project):
    exit_code = main([
        "smoke",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(synthetic_project.results_dir),
        "--attack", "ddos",
    ])
    assert exit_code == 0
    assert (synthetic_project.results_dir / "smoke" / "normal" / "repo_plus.json").exists()
    assert (synthetic_project.results_dir / "smoke" / "adversarial" / "ddos.json").exists()
    assert not (synthetic_project.results_dir / "adversarial" / "primary").exists()


def test_result_and_checkpoint_paths_use_the_canonical_attack_name(tmp_path):
    results_dir = tmp_path / "results"
    expected_result = results_dir / "adversarial" / "primary" / "ddos.json"
    expected_checkpoint = (
        results_dir / "adversarial" / "primary" / "ddos.checkpoint.npz"
    )

    assert result_path(results_dir, "primary", "DDoS") == expected_result
    assert checkpoint_path(results_dir, "primary", "DDoS") == expected_checkpoint


def _adversarial_args(project, *extra: str) -> list[str]:
    return [
        "adversarial",
        "--data-dir", str(project.data_dir),
        "--model-dir", str(project.model_dir),
        "--meta-dir", str(project.meta_dir),
        "--results-dir", str(project.results_dir),
        "--attack", "ddos",
        "--max-samples", "1",
        "--n-masks", "5",
        "--checkpoint-every", "1",
        *extra,
    ]


def _no_change(self, sample, sample_index):
    return AttackOutcome(kind="no_change", x_adv=None, score=0.0)


def test_checkpoint_with_a_different_config_requires_force(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    assert main(_adversarial_args(synthetic_project)) == 0
    result = synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
    first = json.loads(result.read_text())

    # A different seed changes the immutable run parameters, so the existing
    # checkpoint must be rejected instead of silently mixing two runs.
    assert main(_adversarial_args(synthetic_project, "--seed", "7")) != 0
    assert json.loads(result.read_text()) == first

    assert main(_adversarial_args(synthetic_project, "--seed", "7", "--force")) == 0
    forced = json.loads(result.read_text())
    assert forced["config"]["seed"] == 7
    assert forced["counts"]["n"] == 1


def test_attack_all_writes_one_result_per_attack(synthetic_project, monkeypatch):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    exit_code = main(_adversarial_args(synthetic_project, "--attack", "all"))
    assert exit_code == 0

    output_dir = synthetic_project.results_dir / "adversarial" / "primary"
    written = sorted(path.stem for path in output_dir.glob("*.json"))
    assert written == sorted(ATTACK_LABELS)


def test_score_repeats_is_forwarded_to_original_and_victim_scoring(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    real_score_samples = cli.score_samples
    calls: list[dict[str, object]] = []

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_score_samples(*args, **kwargs)

    monkeypatch.setattr(cli, "score_samples", spy)

    exit_code = main(_adversarial_args(synthetic_project, "--score-repeats", "2"))
    assert exit_code == 0

    phases = {call["phase"] for call in calls}
    assert PHASE_ORIGINAL in phases
    assert PHASE_VICTIM in phases
    assert all(call["score_repeats"] == 2 for call in calls)


def test_missing_normal_result_recomputes_and_warns(
    synthetic_project, monkeypatch, caplog
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    synthetic_project.normal_repo_plus_json.unlink()

    with caplog.at_level(logging.WARNING):
        assert main(_adversarial_args(synthetic_project)) == 0

    payload = json.loads(
        (
            synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
        ).read_text()
    )
    assert payload["thresholds"]["fpr_0.10"] >= 0.0
    assert "recomputing" in caplog.text


def test_paper_policy_does_not_reuse_the_calibrated_fpr(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    exit_code = main(
        _adversarial_args(
            synthetic_project,
            "--variant",
            "sensitivity_paper_0.8",
            "--attack-threshold-policy",
            "paper_0.8",
        )
    )
    assert exit_code == 0

    payload = json.loads(
        (
            synthetic_project.results_dir
            / "adversarial"
            / "sensitivity_paper_0.8"
            / "ddos.json"
        ).read_text()
    )
    assert payload["thresholds"]["policy"] == "paper_0.8"
    assert payload["thresholds"]["fpr_0.10"] == pytest.approx(0.123)
    assert payload["thresholds"]["attack"] == pytest.approx(0.8 * 0.123)
    # The benign sample of the fixture has one valid row, so its measured FPR
    # is 0 or 1 -- never the calibrated 0.10 copied from the normal result.
    assert payload["fpr_at_attack_threshold"] in {0.0, 1.0}
