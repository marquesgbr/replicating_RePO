"""Tests for the ``validate`` and ``normal`` CLI entrypoints."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import new_data_eval.cli as cli
from new_data_eval.cli import build_parser, main
from new_data_eval.data import EXPECTED_LABELS
from new_data_eval.results import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]


def _normal_args(project, *extra: str):
    return [
        "normal",
        "--data-dir",
        str(project.data_dir),
        "--model-dir",
        str(project.model_dir),
        "--meta-dir",
        str(project.meta_dir),
        "--results-dir",
        str(project.results_dir),
        "--modes",
        "repo,repo_plus",
        "--n-masks",
        "5",
        "--batch-size",
        "2",
        *extra,
    ]


def test_normal_cli_writes_both_results(synthetic_project):
    exit_code = main(
        [
            "normal",
            "--data-dir",
            str(synthetic_project.data_dir),
            "--model-dir",
            str(synthetic_project.model_dir),
            "--meta-dir",
            str(synthetic_project.meta_dir),
            "--results-dir",
            str(synthetic_project.results_dir),
            "--modes",
            "repo,repo_plus",
            "--n-masks",
            "5",
            "--batch-size",
            "2",
        ]
    )
    assert exit_code == 0
    for mode in ("repo", "repo_plus"):
        payload = json.loads(
            (synthetic_project.results_dir / "normal" / f"{mode}.json").read_text()
        )
        assert payload["experiment"] == "normal"
        assert payload["model_type"] == "new_data_flow"
        assert "thresholds" in payload
        assert "class_rates" in payload


def test_validate_cli_rejects_missing_model(synthetic_project):
    exit_code = main(
        [
            "validate",
            "--data-dir",
            str(synthetic_project.data_dir),
            "--model-dir",
            str(synthetic_project.data_dir / "missing"),
            "--meta-dir",
            str(synthetic_project.meta_dir),
        ]
    )
    assert exit_code != 0


def test_validate_cli_reports_the_dataset_and_constant_features(
    synthetic_project, capsys
):
    exit_code = main(
        [
            "validate",
            "--data-dir",
            str(synthetic_project.data_dir),
            "--model-dir",
            str(synthetic_project.model_dir),
            "--meta-dir",
            str(synthetic_project.meta_dir),
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out.lower()
    assert "constant features" in output
    assert "validation ok" in output


def test_normal_results_carry_the_full_provenance(synthetic_project):
    assert main(_normal_args(synthetic_project, "--max-samples", "4")) == 0

    payload = json.loads(
        (synthetic_project.results_dir / "normal" / "repo_plus.json").read_text()
    )
    assert payload["score_mode"] == "repo_plus"
    assert payload["variant"] == "primary"
    assert payload["dataset"]["name"] == "new_data"
    assert payload["dataset"]["n_samples"] == 4
    assert payload["dataset"]["test_csv_sha256"] == file_sha256(
        synthetic_project.test_csv
    )
    assert payload["scoring"] == {
        "n_masks": 5,
        "score_repeats": 1,
        "paired_scoring": False,
    }
    assert len(payload["config"]["model_sha256"]) == 64
    assert payload["config"]["data_sha256"] == payload["dataset"]["test_csv_sha256"]
    assert payload["environment"]
    assert "git_commit" in payload
    assert payload["runtime_seconds"] >= 0.0


def test_normal_max_samples_stops_after_the_requested_rows(synthetic_project):
    assert main(_normal_args(synthetic_project, "--max-samples", "2")) == 0

    payload = json.loads(
        (synthetic_project.results_dir / "normal" / "repo.json").read_text()
    )
    assert payload["dataset"]["n_samples"] == 2
    assert sum(item["n"] for item in payload["class_rates"]["at_fpr_0.10"].values()) == 2
    # The first two rows of the fixture are valid, so nothing was dropped.
    assert payload["dataset"]["n_dropped_invalid"] == 0


def test_normal_writes_per_sample_scores_and_thresholds(synthetic_project):
    assert main(_normal_args(synthetic_project, "--max-samples", "4")) == 0

    for mode in ("repo", "repo_plus"):
        payload = json.loads(
            (synthetic_project.results_dir / "normal" / f"{mode}.json").read_text()
        )
        score_path = (
            synthetic_project.results_dir / "scores" / f"normal_{mode}.npz"
        )
        with np.load(score_path, allow_pickle=False) as data:
            assert {
                "scores",
                "labels",
                "sample_indices",
                "threshold_fpr_0.01",
                "threshold_fpr_0.10",
            }.issubset(data.files)
            assert data["scores"].shape == (4,)
            assert data["labels"].shape == (4,)
            np.testing.assert_array_equal(
                data["sample_indices"], np.arange(4, dtype=np.int64)
            )
            assert float(data["threshold_fpr_0.01"][0]) == (
                payload["thresholds"]["fpr_0.01"]
            )
            assert float(data["threshold_fpr_0.10"][0]) == (
                payload["thresholds"]["fpr_0.10"]
            )


def test_normal_score_files_are_gitignored():
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "results/new_data/scores/normal_repo_plus.npz",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_normal_results_record_the_normalization_metadata_hashes(synthetic_project):
    assert main(_normal_args(synthetic_project)) == 0

    payload = json.loads(
        (synthetic_project.results_dir / "normal" / "repo_plus.json").read_text()
    )
    assert payload["config"]["meta_sha256"] == {
        "train_min": file_sha256(synthetic_project.meta_dir / "train_min.npy"),
        "train_max": file_sha256(synthetic_project.meta_dir / "train_max.npy"),
    }


def test_normal_result_hash_changes_with_the_normalization_metadata(
    synthetic_project,
):
    assert main(_normal_args(synthetic_project)) == 0
    first = json.loads(
        (synthetic_project.results_dir / "normal" / "repo_plus.json").read_text()
    )

    train_max = np.load(synthetic_project.meta_dir / "train_max.npy")
    np.save(synthetic_project.meta_dir / "train_max.npy", train_max * 1.5)

    assert main(_normal_args(synthetic_project)) == 0
    second = json.loads(
        (synthetic_project.results_dir / "normal" / "repo_plus.json").read_text()
    )

    assert second["config"]["meta_sha256"]["train_max"] != (
        first["config"]["meta_sha256"]["train_max"]
    )
    assert second["config_hash"] != first["config_hash"]


def test_normal_thresholds_are_computed_on_benign_scores(synthetic_project):
    assert main(_normal_args(synthetic_project)) == 0

    for mode in ("repo", "repo_plus"):
        payload = json.loads(
            (synthetic_project.results_dir / "normal" / f"{mode}.json").read_text()
        )
        rates = payload["class_rates"]["at_fpr_0.01"]["BENIGN"]
        # The fixture has one valid BENIGN row (the other two are the invalid
        # rows), so the largest benign score is that row's score and it
        # reaches the threshold.
        assert rates["n"] == 1
        assert rates["fpr"] == 1.0
        assert payload["fpr"]["at_fpr_0.01_threshold"] == 1.0
        assert (
            payload["thresholds"]["fpr_0.01"]
            == payload["thresholds"]["fpr_0.10"]
        )


def test_normal_streaming_is_chunk_size_invariant(synthetic_project):
    assert main(_normal_args(synthetic_project, "--chunk-size", "1")) == 0
    small = json.loads(
        (synthetic_project.results_dir / "normal" / "repo_plus.json").read_text()
    )

    assert main(_normal_args(synthetic_project, "--chunk-size", "7")) == 0
    large = json.loads(
        (synthetic_project.results_dir / "normal" / "repo_plus.json").read_text()
    )

    for key in (
        "thresholds",
        "fpr",
        "class_rates",
        "average_tpr",
        "dataset",
        "scoring",
    ):
        assert small[key] == large[key]


def test_build_parser_exposes_the_documented_defaults():
    parser = build_parser()
    args = parser.parse_args(["normal"])
    assert Path(args.results_dir) == REPO_ROOT / "results" / "new_data"
    assert args.modes == "repo,repo_plus"
    assert args.n_masks == 100
    assert args.batch_size == 100
    assert args.score_repeats == 1
    assert args.seed == 2019
    assert args.max_samples is None


def test_parser_rejects_a_negative_seed():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["normal", "--seed", "-1"])


def test_benign_rows_ignore_the_other_labels(synthetic_frame, tmp_path):
    frame = synthetic_frame(["ddos", "BENIGN", "BENIGN", "xss"], seed=3)
    # Dst Port is the first model feature, so it identifies the returned rows.
    frame["Dst Port"] = [11, 22, 33, 44]
    path = tmp_path / "mixed_train.csv"
    frame.to_csv(path, index=False)

    rows = cli._benign_rows(path)
    assert rows.shape == (2, 77)
    assert sorted(rows[:, 0].tolist()) == [22.0, 33.0]


def test_model_hash_ignores_the_training_manifest(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "variables").mkdir(parents=True)
    (model_dir / "saved_model.pb").write_bytes(b"graph")
    (model_dir / "fingerprint.pb").write_bytes(b"fingerprint")
    (model_dir / "variables" / "variables.index").write_bytes(b"index")

    digest = cli._sha256_path(model_dir)

    manifest = model_dir / "training_manifest.json"
    manifest.write_text('{"created_at": "2026-11-09T10:00:00Z"}')
    assert cli._sha256_path(model_dir) == digest

    manifest.write_text('{"created_at": "2026-11-09T11:00:00Z"}')
    assert cli._sha256_path(model_dir) == digest

    (model_dir / "variables" / "variables.index").write_bytes(b"changed")
    assert cli._sha256_path(model_dir) != digest


def test_write_npz_atomic_fsyncs_the_file_and_the_directory(
    tmp_path, monkeypatch
):
    fsync_calls: list[int] = []
    directory_calls: list[Path] = []
    monkeypatch.setattr(cli.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))
    monkeypatch.setattr(
        cli, "_fsync_directory", lambda path: directory_calls.append(Path(path))
    )

    path = tmp_path / "scores" / "values.npz"
    cli._write_npz_atomic(path, {"values": np.arange(3, dtype=np.float64)})

    assert fsync_calls
    assert directory_calls == [path.parent]
    with np.load(path) as data:
        np.testing.assert_array_equal(data["values"], np.arange(3))


def test_normal_rejects_unknown_modes(synthetic_project):
    assert main(_normal_args(synthetic_project, "--modes", "repo,unknown")) != 0


def test_normal_repo_plus_requires_a_mask_count_divisible_by_five(synthetic_project):
    args = [
        "normal",
        "--data-dir",
        str(synthetic_project.data_dir),
        "--model-dir",
        str(synthetic_project.model_dir),
        "--meta-dir",
        str(synthetic_project.meta_dir),
        "--results-dir",
        str(synthetic_project.results_dir),
        "--modes",
        "repo_plus",
        "--n-masks",
        "3",
    ]
    assert main(args) != 0
    # The fixture pre-creates a placeholder repo_plus.json for the adversarial
    # tests; a failed normal run must not replace it with a result document.
    payload = json.loads(synthetic_project.normal_repo_plus_json.read_text())
    assert "experiment" not in payload


def test_normal_class_rates_cover_every_expected_label(synthetic_project):
    assert main(_normal_args(synthetic_project)) == 0

    payload = json.loads(
        (synthetic_project.results_dir / "normal" / "repo.json").read_text()
    )
    for threshold_name in ("at_fpr_0.01", "at_fpr_0.10"):
        assert set(payload["class_rates"][threshold_name]) == set(EXPECTED_LABELS)
    # The fixture drops two invalid rows (one BENIGN and one attack).
    assert payload["dataset"]["n_dropped_invalid"] == 2
    assert np.isfinite(payload["thresholds"]["fpr_0.01"])
