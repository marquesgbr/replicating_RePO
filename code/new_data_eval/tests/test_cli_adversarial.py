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

import numpy as np
import pandas as pd
import pytest

import new_data_eval.cli as cli
from new_data_eval.attack import AttackOutcome
from new_data_eval.cli import (
    checkpoint_path,
    load_normal_threshold,
    main,
    result_path,
)
from new_data_eval.data import (
    ATTACK_LABELS,
    FEATURE_NAMES,
    integer_feature_indices,
    sha256_file,
)
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
        "--sample-mode", "first",
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

    normal_payload = json.loads(
        (
            synthetic_project.results_dir / "smoke" / "normal" / "repo_plus.json"
        ).read_text()
    )
    assert normal_payload["variant"] == "smoke"
    adversarial_payload = json.loads(
        (
            synthetic_project.results_dir / "smoke" / "adversarial" / "ddos.json"
        ).read_text()
    )
    assert adversarial_payload["variant"] == "smoke"


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


def _rewrite_normal(project, mutate) -> None:
    """Apply ``mutate`` to the fixture's normal result and write it back."""
    payload = json.loads(project.normal_repo_plus_json.read_text())
    mutate(payload)
    project.normal_repo_plus_json.write_text(json.dumps(payload, indent=2))


def _no_change(self, sample, sample_index):
    return AttackOutcome(kind="no_change", x_adv=None, score=0.0)


def test_checkpoint_resume_allows_a_different_batch_size(
    synthetic_project, monkeypatch
):
    """Scores are batch-invariant, so a GPU OOM retry must resume.

    ``--batch-size`` is not part of the checkpoint identity: rerunning with a
    smaller batch must load the existing checkpoint instead of requiring
    ``--force``.
    """
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    args = _adversarial_args(synthetic_project, "--batch-size", "2")
    assert main(args) == 0
    result = synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
    first = json.loads(result.read_text())

    retry = _adversarial_args(synthetic_project, "--batch-size", "4")
    assert main(retry) == 0
    second = json.loads(result.read_text())
    assert second["counts"] == first["counts"]
    assert second["config_hash"] == first["config_hash"]


def test_attack_uses_the_selected_rows_global_indices(
    synthetic_project, monkeypatch
):
    """Masks must be derived from the CSV row indices, not loop positions."""
    seen_indices: list[int] = []

    def record_find_adv(self, sample, sample_index):
        seen_indices.append(int(sample_index))
        return AttackOutcome(kind="no_change", x_adv=None, score=0.0)

    monkeypatch.setattr(cli.FlowAttack, "find_adv", record_find_adv)

    real_score_samples = cli.score_samples
    score_calls: list[np.ndarray] = []

    def spy(*args, **kwargs):
        score_calls.append(np.asarray(kwargs["sample_indices"], dtype=np.int64))
        return real_score_samples(*args, **kwargs)

    monkeypatch.setattr(cli, "score_samples", spy)

    args = _adversarial_args(
        synthetic_project,
        "--max-samples",
        "2",
        "--sample-mode",
        "first",
        "--variant",
        "sensitivity_first_n",
    )
    assert main(args) == 0

    # The fixture places the three ddos rows at global indices 22, 23 and 24.
    expected = np.array([22, 23], dtype=np.int64)
    assert sorted(seen_indices) == expected.tolist()
    assert score_calls
    for indices in score_calls:
        np.testing.assert_array_equal(indices, expected)

    with np.load(
        cli._scores_path(
            synthetic_project.results_dir, "sensitivity_first_n", "ddos"
        )
    ) as data:
        np.testing.assert_array_equal(data["sample_indices"], expected)


@pytest.mark.parametrize("field", ["sample_indices", "n_samples"])
def test_checkpoint_rejects_a_changed_sampling(
    synthetic_project, monkeypatch, capsys, field
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    args = _adversarial_args(synthetic_project)
    assert main(args) == 0

    checkpoint = checkpoint_path(synthetic_project.results_dir, "primary", "ddos")
    with np.load(checkpoint) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    if field == "sample_indices":
        arrays[field] = np.asarray(arrays[field], dtype=np.int64) + 1
    else:
        arrays[field] = np.asarray([2], dtype=np.int64)
    np.savez_compressed(checkpoint, **arrays)

    assert main(args) != 0
    assert "sampling changed" in capsys.readouterr().err


def test_smoke_force_is_forwarded():
    parser = cli.build_parser()
    assert parser.parse_args(["smoke"]).force is False
    assert parser.parse_args(["smoke", "--force"]).force is True


def test_stable_run_config_excludes_batch_size_and_includes_the_commit():
    config = cli._stable_run_config(
        attack="ddos",
        variant="primary",
        seed=2019,
        sample_mode="random",
        max_samples=1,
        n_masks=5,
        score_repeats=1,
        paired_scoring=False,
        policy="calibrated",
        real_threshold=0.5,
        attack_threshold=0.5,
        model_sha256="a" * 64,
        data_sha256="b" * 64,
        meta_sha256={"train_min": "c" * 64, "train_max": "d" * 64},
        normal_result={"used": True},
        git_commit="e" * 40,
    )
    assert "batch_size" not in config
    assert config["git_commit"] == "e" * 40

    other_commit = dict(config)
    other_commit["git_commit"] = "f" * 40
    assert cli.config_hash(config) != cli.config_hash(other_commit)


def test_checkpoint_with_a_different_config_requires_force(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    assert main(_adversarial_args(synthetic_project)) == 0
    result = synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
    first = json.loads(result.read_text())

    # A different max_samples changes the immutable run parameters, so the
    # existing checkpoint must be rejected instead of silently mixing two runs.
    assert (
        main(_adversarial_args(synthetic_project, "--max-samples", "2")) != 0
    )
    assert json.loads(result.read_text()) == first

    assert (
        main(
            _adversarial_args(
                synthetic_project, "--max-samples", "2", "--force"
            )
        )
        == 0
    )
    forced = json.loads(result.read_text())
    assert forced["config"]["max_samples"] == 2
    assert forced["counts"]["n"] == 2


def test_force_does_not_bypass_the_normal_provenance(
    synthetic_project, monkeypatch, capsys
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    _rewrite_normal(
        synthetic_project,
        lambda payload: payload["config"].update({"seed": 123}),
    )

    assert main(_adversarial_args(synthetic_project, "--force")) != 0
    assert "config.seed" in capsys.readouterr().err


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
    _rewrite_normal(
        synthetic_project,
        lambda payload: payload["scoring"].update({"score_repeats": 2}),
    )
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


def test_calibrated_fallback_measures_the_fpr_on_benign_rows(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    synthetic_project.normal_repo_plus_json.unlink()

    assert main(_adversarial_args(synthetic_project)) == 0

    payload = json.loads(
        (
            synthetic_project.results_dir / "adversarial" / "primary" / "ddos.json"
        ).read_text()
    )
    assert payload["thresholds"]["policy"] == "calibrated"
    # The fixture has one valid benign row, so the measured FPR is 0 or 1 --
    # never the nominal 0.10 that the old fallback fabricated.
    assert payload["fpr_at_attack_threshold"] in {0.0, 1.0}
    assert payload["fpr_at_attack_threshold"] != pytest.approx(0.10)


def test_primary_rejects_a_truncated_normal_result(
    synthetic_project, monkeypatch, capsys
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    _rewrite_normal(
        synthetic_project,
        lambda payload: payload["config"]["sampling"].update(
            {"mode": "first_n"}
        ),
    )

    assert main(_adversarial_args(synthetic_project)) != 0
    assert "config.sampling.mode" in capsys.readouterr().err
    assert not result_path(
        synthetic_project.results_dir, "primary", "ddos"
    ).exists()


def test_smoke_variant_accepts_a_truncated_normal_result(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    _rewrite_normal(
        synthetic_project,
        lambda payload: payload["config"]["sampling"].update(
            {"mode": "first_n"}
        ),
    )

    assert main(
        _adversarial_args(synthetic_project, "--variant", "smoke")
    ) == 0

    smoke_result = result_path(
        synthetic_project.results_dir, "smoke", "ddos"
    )
    assert smoke_result.exists()
    assert not result_path(
        synthetic_project.results_dir, "primary", "ddos"
    ).exists()
    assert json.loads(smoke_result.read_text())["variant"] == "smoke"


def test_smoke_variant_cannot_overwrite_primary(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    assert main(_adversarial_args(synthetic_project)) == 0
    primary = result_path(synthetic_project.results_dir, "primary", "ddos")
    original = primary.read_bytes()

    assert main(
        _adversarial_args(synthetic_project, "--variant", "smoke")
    ) == 0

    assert primary.read_bytes() == original
    assert result_path(
        synthetic_project.results_dir, "smoke", "ddos"
    ).exists()


@pytest.mark.parametrize(
    "variant, sample_mode, policy",
    [
        ("primary", "first", "calibrated"),
        ("primary", "random", "paper_0.8"),
        ("sensitivity_first_n", "random", "calibrated"),
        ("sensitivity_first_n", "first", "paper_0.8"),
        ("sensitivity_paper_0.8", "random", "calibrated"),
        ("sensitivity_paper_0.8", "first", "paper_0.8"),
    ],
)
def test_variant_rejects_an_inconsistent_policy_or_sample_mode(
    synthetic_project, monkeypatch, variant, sample_mode, policy
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    exit_code = main(
        _adversarial_args(
            synthetic_project,
            "--variant",
            variant,
            "--sample-mode",
            sample_mode,
            "--attack-threshold-policy",
            policy,
        )
    )
    assert exit_code != 0
    assert not result_path(
        synthetic_project.results_dir, variant, "ddos"
    ).exists()


def test_invalid_variant_cannot_overwrite_primary(synthetic_project, monkeypatch):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    assert main(_adversarial_args(synthetic_project)) == 0
    primary = result_path(synthetic_project.results_dir, "primary", "ddos")
    original = primary.read_bytes()

    exit_code = main(
        _adversarial_args(synthetic_project, "--sample-mode", "first")
    )
    assert exit_code != 0
    assert primary.read_bytes() == original


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda payload: payload["config"].update({"model_sha256": "0" * 64}),
            "model_sha256",
        ),
        (
            lambda payload: payload["dataset"].update(
                {"test_csv_sha256": "0" * 64}
            ),
            "test_csv_sha256",
        ),
        (
            lambda payload: payload.update({"model_type": "flow"}),
            "model_type",
        ),
        (
            lambda payload: payload["config"].update({"seed": 7}),
            "config.seed",
        ),
        (
            lambda payload: payload["config"]["sampling"].update(
                {"mode": "first_n"}
            ),
            "sampling.mode",
        ),
        (
            lambda payload: payload["config"]["meta_sha256"].update(
                {"train_min": "0" * 64}
            ),
            "meta_sha256.train_min",
        ),
        (
            lambda payload: payload.update({"variant": "smoke"}),
            "variant",
        ),
        (
            lambda payload: payload.update({"score_mode": "repo"}),
            "score_mode",
        ),
    ],
)
def test_adversarial_rejects_a_normal_result_from_another_run(
    synthetic_project, monkeypatch, capsys, mutate, match
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)
    _rewrite_normal(synthetic_project, mutate)

    exit_code = main(_adversarial_args(synthetic_project))
    assert exit_code != 0
    assert match in capsys.readouterr().err
    assert not result_path(
        synthetic_project.results_dir, "primary", "ddos"
    ).exists()
    assert not cli._scores_path(
        synthetic_project.results_dir, "primary", "ddos"
    ).exists()


def test_adversarial_rejects_mismatched_n_masks(
    synthetic_project, monkeypatch, capsys
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    exit_code = main(_adversarial_args(synthetic_project, "--n-masks", "10"))
    assert exit_code != 0
    assert "n_masks" in capsys.readouterr().err
    assert not result_path(
        synthetic_project.results_dir, "primary", "ddos"
    ).exists()


def test_adversarial_records_the_normal_result_provenance(
    synthetic_project, monkeypatch
):
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    assert main(_adversarial_args(synthetic_project)) == 0

    payload = json.loads(
        result_path(synthetic_project.results_dir, "primary", "ddos").read_text()
    )
    normal = payload["config"]["normal_result"]
    assert Path(normal["path"]).name == "repo_plus.json"
    assert normal["config_hash"] == "a" * 64
    assert normal["used"] is True


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


def test_decimal_features_are_derived_from_the_full_test_set(
    synthetic_project, monkeypatch
):
    """A feature integer only inside the attack subset must stay fractional.

    The reference derives ``decimal_features`` from the concatenated valid
    test matrix.  If the CLI instead derived it from the selected attack rows,
    this feature would be rounded by the attack and would change the
    adversarial projection.
    """
    column = "Tot Fwd Pkts"
    feature_index = FEATURE_NAMES.index(column)
    frame = pd.read_csv(synthetic_project.test_csv)
    is_ddos = frame["Label"].str.lower() == "ddos"
    assert is_ddos.any()
    frame.loc[is_ddos, column] = 7.0
    frame.loc[~is_ddos, column] = 7.5
    frame.to_csv(synthetic_project.test_csv, index=False)
    _rewrite_normal(
        synthetic_project,
        lambda payload: payload["dataset"].update(
            {"test_csv_sha256": sha256_file(synthetic_project.test_csv)}
        ),
    )

    expected = integer_feature_indices(synthetic_project.test_csv)
    assert feature_index not in expected

    seen: list[list[int]] = []
    original_init = cli.FlowAttack.__init__

    def spy_init(self, *args, **kwargs):
        seen.append(list(args[3]))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(cli.FlowAttack, "__init__", spy_init)
    monkeypatch.setattr(cli.FlowAttack, "find_adv", _no_change)

    assert main(_adversarial_args(synthetic_project)) == 0
    assert seen
    assert all(features == expected for features in seen)
    assert all(feature_index not in features for features in seen)


def test_interrupted_checkpoint_resumes_and_matches_a_clean_run(
    synthetic_project, monkeypatch
):
    """Stop after one sample, resume, and lock every reported value."""
    results_dir = synthetic_project.results_dir
    base_args = [
        "adversarial",
        "--data-dir", str(synthetic_project.data_dir),
        "--model-dir", str(synthetic_project.model_dir),
        "--meta-dir", str(synthetic_project.meta_dir),
        "--results-dir", str(results_dir),
        "--attack", "ddos",
        "--max-samples", "2",
        "--n-masks", "5",
        "--checkpoint-every", "1",
    ]
    state = {"calls": 0, "stop": True}

    def stop_after_one_sample():
        if not state["stop"]:
            return False
        state["calls"] += 1
        return state["calls"] >= 1

    monkeypatch.setattr(cli, "stop_requested", stop_after_one_sample)

    assert main(base_args) == cli.INTERRUPTED_EXIT_CODE
    intermediate = checkpoint_path(results_dir, "primary", "ddos")
    assert intermediate.is_file()
    assert not result_path(results_dir, "primary", "ddos").exists()
    with np.load(intermediate) as data:
        assert int(np.asarray(data["next_index"]).ravel()[0]) == 1
        # The interrupted run has not scored the victims yet; resuming must
        # recompute them instead of reusing these zeros.
        assert np.all(np.asarray(data["victim_scores"]) == 0.0)

    state["stop"] = False
    assert main(base_args) == 0

    resumed = json.loads(
        result_path(results_dir, "primary", "ddos").read_text()
    )
    with np.load(cli._scores_path(results_dir, "primary", "ddos")) as data:
        resumed_scores = {
            field: np.array(data[field], copy=True)
            for field in ("original_scores", "victim_scores", "kinds")
        }

    # A clean run with the same sampling is the reference for every count,
    # rate, score and kind; --force discards the checkpoint and the result.
    assert main([*base_args, "--force"]) == 0
    clean = json.loads(result_path(results_dir, "primary", "ddos").read_text())
    for field in (
        "counts",
        "victim_tpr",
        "normal_tpr_at_attack_threshold",
    ):
        assert resumed[field] == clean[field]

    with np.load(cli._scores_path(results_dir, "primary", "ddos")) as clean_scores:
        for field, resumed_values in resumed_scores.items():
            np.testing.assert_array_equal(
                resumed_values, clean_scores[field], err_msg=field
            )
