"""Tests for the dataset extraction/validation layer."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from new_data_eval.data import (
    ATTACK_LABELS,
    EXPECTED_COLUMNS,
    EXPECTED_LABELS,
    FEATURE_INDICES,
    LABEL_INDEX,
    N_FEATURES,
    canonical_label,
    count_classes,
    feature_frame,
    iter_test_chunks,
    load_metadata,
    normalize_features,
    select_attack_samples,
    validate_dataset,
    validate_header,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_canonical_label_normalizes_benign_and_attack():
    assert canonical_label("  benign ") == "BENIGN"
    assert canonical_label("DDoS") == "ddos"
    assert canonical_label(" xss ") == "xss"


def test_validate_header_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="85"):
        validate_header(["column00", "Flow ID", "Label"])


def test_validate_header_rejects_wrong_column_order():
    columns = list(EXPECTED_COLUMNS)
    columns[5], columns[6] = columns[6], columns[5]
    with pytest.raises(ValueError, match="column"):
        validate_header(columns)


def test_feature_selection_returns_77_columns(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv, nrows=1)
    x, labels, valid = feature_frame(frame)
    assert x.shape == (1, 77)
    assert labels[0] == "BENIGN"
    assert valid[0]


def test_feature_frame_rejects_a_full_width_frame_with_wrong_order(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv, nrows=2)
    columns = list(EXPECTED_COLUMNS)
    columns[5], columns[6] = columns[6], columns[5]

    with pytest.raises(ValueError, match="column"):
        feature_frame(frame.loc[:, columns])


def test_feature_frame_selects_a_narrow_chunk_frame_by_name(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv, nrows=2)
    from new_data_eval.data import FEATURE_NAMES

    narrow = frame.loc[:, [*FEATURE_NAMES, "Label"]]
    x, labels, valid = feature_frame(narrow)
    assert x.shape == (2, N_FEATURES)
    assert list(labels) == ["BENIGN", "xss"]
    assert valid.all()


def test_invalid_rows_are_dropped_and_counted(synthetic_project):
    info = validate_dataset(
        synthetic_project.train_csv,
        synthetic_project.test_csv,
        synthetic_project.meta_dir,
    )
    assert info.test_dropped_invalid == 2


def test_validate_dataset_rejects_unknown_label(synthetic_project):
    with pytest.raises(ValueError, match="not_an_attack"):
        validate_dataset(
            synthetic_project.train_csv,
            synthetic_project.unknown_label_csv,
            synthetic_project.meta_dir,
        )


def test_select_attack_samples_is_seeded(synthetic_project):
    a, n_a = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2,
        mode="random", seed=2019,
    )
    b, n_b = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2,
        mode="random", seed=2019,
    )
    assert n_a == n_b
    np.testing.assert_allclose(a, b)


# --- layout -----------------------------------------------------------------


def test_expected_columns_describe_the_canonical_layout():
    assert len(EXPECTED_COLUMNS) == 85
    assert EXPECTED_COLUMNS[0] == "column00"
    assert EXPECTED_COLUMNS[5] == "Dst Port"
    assert EXPECTED_COLUMNS[84] == "Label"
    assert LABEL_INDEX == 84
    assert N_FEATURES == 77
    assert len(FEATURE_INDICES) == N_FEATURES
    assert FEATURE_INDICES[0] == 5
    assert set(FEATURE_INDICES) == set(range(8, 84)) | {5}


def test_expected_labels_cover_benign_and_nine_attacks():
    assert EXPECTED_LABELS[0] == "BENIGN"
    assert len(ATTACK_LABELS) == 9
    assert set(ATTACK_LABELS) == {
        "xss",
        "password",
        "injection",
        "scanning",
        "backdoor",
        "ransomware",
        "mitm",
        "ddos",
        "dos",
    }


def test_validate_header_accepts_the_canonical_header():
    validate_header(list(EXPECTED_COLUMNS))


def test_canonical_label_rejects_unknown_labels():
    with pytest.raises(ValueError, match="not_an_attack"):
        canonical_label("not_an_attack")


# --- metadata and normalization ---------------------------------------------


def test_load_metadata_returns_77_element_arrays(synthetic_project):
    train_min, train_max = load_metadata(synthetic_project.meta_dir)
    assert train_min.shape == (77,)
    assert train_max.shape == (77,)
    assert np.isfinite(train_min).all()
    assert np.isfinite(train_max).all()
    assert (train_max >= train_min).all()


def test_load_metadata_rejects_wrong_shape(synthetic_project):
    np.save(synthetic_project.meta_dir / "train_min.npy", np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="77"):
        load_metadata(synthetic_project.meta_dir)


def test_normalize_features_uses_the_training_bounds():
    x = np.arange(77, dtype=np.float32).reshape(1, 77)
    train_min = np.zeros(77, dtype=np.float32)
    train_max = np.full(77, 4.0, dtype=np.float32)
    expected = (x - train_min) / (train_max - train_min + 1e-6)
    np.testing.assert_allclose(
        normalize_features(x, train_min, train_max), expected, rtol=1e-6, atol=1e-6
    )


def test_normalize_features_rejects_wrong_width():
    with pytest.raises(ValueError, match="77"):
        normalize_features(np.zeros((2, 76), dtype=np.float32), np.zeros(77), np.ones(77))


# --- CSV loading ------------------------------------------------------------


def test_feature_frame_marks_non_finite_rows_as_invalid(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv)
    x, labels, valid = feature_frame(frame)
    assert x.shape == (30, 77)
    assert len(labels) == 30
    assert int(valid.sum()) == 28
    assert int((~valid).sum()) == 2


def test_iter_test_chunks_drops_invalid_rows(synthetic_project):
    total = 0
    for x, labels in iter_test_chunks(synthetic_project.test_csv, chunksize=7):
        assert x.shape[1] == N_FEATURES
        assert len(x) == len(labels)
        total += len(x)
    assert total == 28


def test_iter_test_chunks_is_chunk_size_invariant(synthetic_project):
    small = np.concatenate([chunk for chunk, _ in iter_test_chunks(synthetic_project.test_csv, chunksize=7)])
    large = np.concatenate([chunk for chunk, _ in iter_test_chunks(synthetic_project.test_csv, chunksize=64)])
    assert small.shape == large.shape
    np.testing.assert_allclose(small, large)


def test_count_classes_counts_only_valid_rows(synthetic_project):
    counts = count_classes(synthetic_project.test_csv)
    assert counts["BENIGN"] == 1
    assert counts["ddos"] == 3
    assert sum(counts.values()) == 28


# --- sampling ---------------------------------------------------------------


def test_select_attack_samples_first_mode_takes_the_prefix(synthetic_project):
    x, available = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2, mode="first", seed=2019
    )
    assert available == 3
    assert x.shape == (2, N_FEATURES)


def test_select_attack_samples_is_case_insensitive(synthetic_project):
    x, available = select_attack_samples(
        synthetic_project.test_csv, "DDoS", max_samples=2, mode="random", seed=2019
    )
    assert available == 3
    assert x.shape == (2, N_FEATURES)


def test_select_attack_samples_is_chunk_size_invariant(synthetic_project):
    small, n_small = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2, mode="random",
        seed=2019, chunksize=7,
    )
    large, n_large = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2, mode="random",
        seed=2019, chunksize=64,
    )
    assert n_small == n_large == 3
    np.testing.assert_allclose(small, large)


def test_select_attack_samples_changes_with_the_seed(synthetic_project):
    first, _ = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2, mode="random", seed=2019
    )
    second, _ = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=2, mode="random", seed=7
    )
    assert not np.array_equal(first, second)


def test_select_attack_samples_random_mode_is_not_the_prefix(
    synthetic_frame, tmp_path
):
    labels = ["ddos"] * 40 + ["xss"] * 10
    csv_path = tmp_path / "single_class.csv"
    synthetic_frame(labels, seed=5).to_csv(csv_path, index=False)

    random_a, available = select_attack_samples(
        csv_path, "ddos", max_samples=5, mode="random", seed=2019
    )
    random_b, _ = select_attack_samples(
        csv_path, "ddos", max_samples=5, mode="random", seed=2020
    )
    prefix, _ = select_attack_samples(
        csv_path, "ddos", max_samples=5, mode="first", seed=2019
    )

    assert available == 40
    assert random_a.shape == (5, N_FEATURES)
    # The random mode is neither the prefix nor a function of the chunk size.
    assert not np.array_equal(random_a, prefix)
    assert not np.array_equal(random_a, random_b)
    chunked, _ = select_attack_samples(
        csv_path, "ddos", max_samples=5, mode="random", seed=2019, chunksize=7
    )
    np.testing.assert_allclose(random_a, chunked)


def test_select_attack_samples_keeps_the_last_draw_per_slot(
    synthetic_frame, tmp_path
):
    """The reservoir must not depend on NumPy's repeated-index assignment."""
    labels = ["ddos"] * 40
    csv_path = tmp_path / "single_class.csv"
    synthetic_frame(labels, seed=5).to_csv(csv_path, index=False)

    selected, _ = select_attack_samples(
        csv_path, "ddos", max_samples=5, mode="random", seed=2019, chunksize=7
    )
    repeated, _ = select_attack_samples(
        csv_path, "ddos", max_samples=5, mode="random", seed=2019, chunksize=64
    )
    np.testing.assert_allclose(selected, repeated)


def test_select_attack_samples_returns_every_row_when_not_truncated(synthetic_project):
    x, available = select_attack_samples(
        synthetic_project.test_csv, "ddos", max_samples=100, mode="random", seed=2019
    )
    assert available == 3
    assert x.shape == (3, N_FEATURES)


def test_select_attack_samples_rejects_unknown_attack(synthetic_project):
    with pytest.raises(ValueError, match="not_an_attack"):
        select_attack_samples(
            synthetic_project.test_csv, "not_an_attack", max_samples=2, mode="random", seed=2019
        )


def test_select_attack_samples_rejects_unknown_mode(synthetic_project):
    with pytest.raises(ValueError, match="mode"):
        select_attack_samples(
            synthetic_project.test_csv, "ddos", max_samples=2, mode="middle", seed=2019
        )


# --- dataset validation -----------------------------------------------------


def test_validate_dataset_reports_rows_hashes_and_counts(synthetic_project):
    info = validate_dataset(
        synthetic_project.train_csv,
        synthetic_project.test_csv,
        synthetic_project.meta_dir,
    )
    assert info.train_rows == 10
    assert info.test_rows == 30
    assert info.test_valid_rows == 28
    assert info.test_dropped_invalid == 2
    assert len(info.train_sha256) == 64
    assert len(info.test_sha256) == 64
    assert info.class_counts["BENIGN"] == 1
    assert info.class_counts["ddos"] == 3


def test_validate_dataset_rejects_non_benign_training_rows(synthetic_project):
    from new_data_eval.data import EXPECTED_COLUMNS

    frame = pd.read_csv(synthetic_project.train_csv)
    frame.loc[0, "Label"] = "ddos"
    broken_train = synthetic_project.data_dir / "broken_train.csv"
    frame.to_csv(broken_train, index=False)

    with pytest.raises(ValueError, match="ddos"):
        validate_dataset(broken_train, synthetic_project.test_csv, synthetic_project.meta_dir)
    assert EXPECTED_COLUMNS[84] == "Label"


def test_validate_dataset_rejects_a_missing_attack_class(synthetic_project):
    from new_data_eval.data import ATTACK_LABELS

    frame = pd.read_csv(synthetic_project.test_csv)
    frame = frame[frame["Label"] != ATTACK_LABELS[-1]]
    reduced = synthetic_project.data_dir / "reduced_test.csv"
    frame.to_csv(reduced, index=False)

    with pytest.raises(ValueError, match=ATTACK_LABELS[-1]):
        validate_dataset(synthetic_project.train_csv, reduced, synthetic_project.meta_dir)


def test_validate_dataset_rejects_a_test_csv_without_benign(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv)
    frame = frame[frame["Label"] != "BENIGN"]
    without_benign = synthetic_project.data_dir / "without_benign.csv"
    frame.to_csv(without_benign, index=False)

    with pytest.raises(ValueError, match="BENIGN"):
        validate_dataset(
            synthetic_project.train_csv, without_benign, synthetic_project.meta_dir
        )


def test_validate_dataset_reports_constant_features(synthetic_project):
    train_min = np.zeros(77, dtype=np.float32)
    train_max = np.full(77, 10.0, dtype=np.float32)
    train_min[3] = 5.0
    train_max[3] = 5.0
    np.save(synthetic_project.meta_dir / "train_min.npy", train_min)
    np.save(synthetic_project.meta_dir / "train_max.npy", train_max)

    info = validate_dataset(
        synthetic_project.train_csv,
        synthetic_project.test_csv,
        synthetic_project.meta_dir,
    )
    assert info.constant_features == (3,)


def test_validate_dataset_reports_no_constant_features_by_default(synthetic_project):
    info = validate_dataset(
        synthetic_project.train_csv,
        synthetic_project.test_csv,
        synthetic_project.meta_dir,
    )
    assert info.constant_features == ()


def test_constant_feature_indices_lists_the_constant_features():
    from new_data_eval.data import constant_feature_indices

    train_min = np.zeros(4, dtype=np.float32)
    train_max = np.array([1.0, 0.0, 3.0, 2.0], dtype=np.float32)
    assert constant_feature_indices(train_min, train_max) == [1]


def test_validate_dataset_rejects_a_wrong_header(synthetic_project):
    frame = pd.read_csv(synthetic_project.test_csv)
    frame = frame.rename(columns={"Dst Port": "Dst Port "})
    broken = synthetic_project.data_dir / "broken_test.csv"
    frame.to_csv(broken, index=False)

    with pytest.raises(ValueError, match="column"):
        validate_dataset(synthetic_project.train_csv, broken, synthetic_project.meta_dir)


# --- repository hygiene -----------------------------------------------------


def test_extracted_csv_files_are_gitignored():
    if shutil.which("git") is None:  # pragma: no cover - git is always present here
        pytest.skip("git is not available")
    result = subprocess.run(
        ["git", "check-ignore", "data/new_data/new_test.csv", "data/new_data/new_train.csv"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_extract_script_fails_with_a_clear_message_without_git_lfs(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    shim = fake_bin / "git"
    shim.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    shim.chmod(0o755)

    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is always present here
        pytest.skip("bash is not available")

    env = {"PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(tmp_path)}
    result = subprocess.run(
        [bash, str(REPO_ROOT / "extract_new_data.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "git-lfs" in combined
    assert os.path.isdir(fake_bin)
