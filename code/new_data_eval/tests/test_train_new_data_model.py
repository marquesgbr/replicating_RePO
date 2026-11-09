"""Focused tests of the training script parser and manifest helper."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import train_new_data_model


def test_training_parser_exposes_reproducibility_options():
    parser = train_new_data_model.build_parser()
    args = parser.parse_args(
        [
            "--train-csv",
            "train.csv",
            "--output-dir",
            "out",
            "--seed",
            "7",
            "--epochs",
            "2",
            "--batch-size",
            "8",
        ]
    )
    assert args.train_csv == "train.csv"
    assert args.output_dir == "out"
    assert args.seed == 7
    assert args.epochs == 2
    assert args.batch_size == 8


def test_training_parser_defaults_match_the_reference_training():
    args = train_new_data_model.build_parser().parse_args([])
    assert args.seed == 2019
    assert args.epochs == 5
    assert args.batch_size == 256


def test_training_manifest_records_hash_versions_and_shape(tmp_path):
    train_csv = tmp_path / "new_train.csv"
    train_csv.write_text("a,b\n1,2\n", encoding="utf-8")
    expected_hash = hashlib.sha256(train_csv.read_bytes()).hexdigest()

    manifest = train_new_data_model.build_training_manifest(
        train_csv,
        seed=2019,
        epochs=5,
        batch_size=256,
        final_loss=0.123,
        shape=(10, 77),
        versions={"tensorflow": "test", "keras": "test"},
    )

    assert manifest["data_sha256"] == expected_hash
    assert manifest["seed"] == 2019
    assert manifest["epochs"] == 5
    assert manifest["batch_size"] == 256
    assert manifest["final_loss"] == pytest.approx(0.123)
    assert manifest["shape"] == [10, 77]
    assert manifest["versions"]["tensorflow"] == "test"


def test_training_manifest_records_the_installed_tensorflow_version(tmp_path):
    train_csv = tmp_path / "new_train.csv"
    train_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    manifest = train_new_data_model.build_training_manifest(
        train_csv,
        seed=2019,
        epochs=1,
        batch_size=2,
        final_loss=0.1,
        shape=(1, 77),
    )

    assert manifest["versions"]["python"]
    assert manifest["versions"]["numpy"]
    assert manifest["versions"]["tensorflow"]
    assert manifest["versions"]["keras"]


def test_training_labels_must_be_benign():
    train_new_data_model.assert_benign_labels(np.array(["BENIGN", "BENIGN"]))
    with pytest.raises(ValueError, match="BENIGN"):
        train_new_data_model.assert_benign_labels(np.array(["BENIGN", "ddos"]))
