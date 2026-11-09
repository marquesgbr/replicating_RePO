"""Command line interface for the new dataset evaluation.

The package is invoked as ``PYTHONPATH=code python -m new_data_eval``.  Only
the ``validate`` and ``normal`` commands live here for now; the adversarial
command is added later on top of the same result builders.

``validate`` checks the canonical header, the labels, the normalization
metadata and the model.  The model check reuses ``validate_model``, which
compares the reconstruction error of **real benign training rows** with the
error of uniform noise.

``normal`` streams ``new_test.csv`` in chunks, scores every valid row with RePO
and RePO+ (using the global row offset as ``sample_indices`` so the masks never
restart inside a chunk), computes the FPR 1% and 10% thresholds on the benign
scores, and writes one JSON document per score mode under
``<results-dir>/normal/``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from new_data_eval import __version__
from new_data_eval.data import (
    BENIGN_LABEL,
    DEFAULT_CHUNK_SIZE,
    iter_test_chunks_detailed,
    load_metadata,
    sha256_file,
    validate_dataset,
)
from new_data_eval.results import (
    build_normal_result,
    environment_manifest,
    file_sha256,
    git_commit,
    write_json_atomic,
)
from new_data_eval.scoring import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_MASKS,
    PHASE_ORIGINAL,
    REPO_PLUS_GROUPS,
    load_model,
    reconstruction_errors,
    score_from_errors,
    score_samples,
    threshold_at_fpr,
    validate_model,
)

__all__ = [
    "build_parser",
    "main",
    "run_normal",
    "run_validate",
]


#: Repository root, used for the default data/model/results directories.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default directories of the new dataset evaluation.
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "new_data"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "new_data_model"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "new_data"

#: Thresholds reported by the normal evaluation, calibrated on BENIGN scores.
NORMAL_THRESHOLD_FPRS: tuple[tuple[str, float], ...] = (
    ("fpr_0.01", 0.01),
    ("fpr_0.10", 0.10),
)

#: Number of real benign training rows fed to ``validate_model``.
VALIDATION_BENIGN_ROWS = 16

#: Chunk size used to read the benign rows during validation (small on purpose).
VALIDATION_CHUNK_SIZE = 1024

#: Score modes supported by the normal evaluation.
SCORE_MODES: tuple[str, ...] = ("repo", "repo_plus")


def _positive_int(value: str) -> int:
    """Return ``value`` as a positive integer for an argparse option."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {number}")
    return number


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the path and randomness options shared by every command."""
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="directory holding new_train.csv and new_test.csv",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="SavedModel directory or Keras model file",
    )
    parser.add_argument(
        "--meta-dir",
        default=None,
        help="directory holding train_min.npy and train_max.npy "
        "(default: <data-dir>/x_train_meta)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="root directory of the result documents",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2019,
        help="seed of the stateless per-sample masks",
    )
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help="rows read per CSV chunk",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of ``python -m new_data_eval``."""
    parser = argparse.ArgumentParser(
        prog="python -m new_data_eval",
        description="Validate the new dataset and run the RePO/RePO+ evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="check the header, labels, metadata and model"
    )
    _add_common_arguments(validate)

    normal = subparsers.add_parser(
        "normal", help="score the test set with RePO and RePO+"
    )
    _add_common_arguments(normal)
    normal.add_argument(
        "--modes",
        default="repo,repo_plus",
        help="comma separated score modes (repo, repo_plus)",
    )
    normal.add_argument(
        "--n-masks",
        type=_positive_int,
        default=DEFAULT_N_MASKS,
        help="number of masks per sample (tests/smoke only)",
    )
    normal.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="masked rows sent to the model per forward pass",
    )
    normal.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="stop after this many valid samples (smoke runs only)",
    )
    normal.add_argument(
        "--score-repeats",
        type=_positive_int,
        default=1,
        help="independent mask repeats whose scores are averaged",
    )
    normal.add_argument(
        "--paired-scoring",
        action="store_true",
        help="use the paired mask seed derivation",
    )

    return parser


def _resolved_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Return ``(data_dir, model_dir, meta_dir, results_dir)`` as paths."""
    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    meta_dir = (
        Path(args.meta_dir).expanduser()
        if args.meta_dir is not None
        else data_dir / "x_train_meta"
    )
    results_dir = Path(args.results_dir).expanduser()
    return data_dir, model_dir, meta_dir, results_dir


def _display_path(path: Path) -> str:
    """Return ``path`` relative to the repository root when possible."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sha256_path(path: Path) -> str:
    """Return the stable SHA-256 of a model file or directory.

    For a directory the relative path of every file is hashed first, so the
    digest depends on the file names and their contents but not on the order in
    which the filesystem lists them.
    """
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Model not found: {path}")

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Model directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _benign_rows(train_csv: Path, max_rows: int = VALIDATION_BENIGN_ROWS) -> np.ndarray:
    """Return up to ``max_rows`` valid BENIGN rows of the training CSV."""
    collected: list[np.ndarray] = []
    count = 0
    for chunk in iter_test_chunks_detailed(train_csv, VALIDATION_CHUNK_SIZE):
        rows = chunk.valid_features
        if len(rows):
            take = rows[: max_rows - count]
            collected.append(take)
            count += len(take)
        if count >= max_rows:
            break
    if not collected:
        raise ValueError(f"No valid benign rows found in {train_csv}")
    return np.concatenate(collected, axis=0)


def _parse_modes(value: str) -> list[str]:
    """Return the canonical score modes listed in ``value``."""
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--modes must list at least one score mode")
    unknown = [mode for mode in modes if mode not in SCORE_MODES]
    if unknown:
        raise ValueError(
            f"unknown score mode(s) {unknown}; expected one of {list(SCORE_MODES)}"
        )
    # Preserve the requested order but drop duplicates.
    return list(dict.fromkeys(modes))


def _collect_scores(
    test_csv: Path,
    model: Any,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    modes: Sequence[str],
    n_masks: int,
    batch_size: int,
    seed: int,
    score_repeats: int,
    paired_seed: int | None,
    chunk_size: int,
    max_samples: int | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    """Stream ``test_csv`` and return the scores, labels and row counts.

    The global row offset of every valid row is passed as ``sample_indices``,
    so the masks of a sample are the same whatever the chunk size is.  When
    ``max_samples`` is set, the reader stops after that many valid rows and the
    invalid rows after the last consumed row are not counted.
    """
    score_parts: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    label_parts: list[np.ndarray] = []
    collected = 0
    dropped = 0

    for chunk in iter_test_chunks_detailed(test_csv, chunk_size):
        if max_samples is not None and collected >= max_samples:
            break

        valid_positions = np.flatnonzero(chunk.valid)
        if len(valid_positions) == 0:
            dropped += len(chunk.valid)
            continue

        if max_samples is None or len(valid_positions) <= max_samples - collected:
            positions = valid_positions
            cutoff = len(chunk.valid)
        else:
            positions = valid_positions[: max_samples - collected]
            cutoff = int(positions[-1]) + 1

        dropped += int(np.count_nonzero(~chunk.valid[:cutoff]))

        x_raw = chunk.features[positions]
        chunk_labels = chunk.labels[positions]
        # Global row indices -- not positions inside the chunk.
        sample_indices = chunk.indices[positions]

        if score_repeats == 1:
            errors = reconstruction_errors(
                model,
                x_raw,
                train_min,
                train_max,
                n_masks=n_masks,
                batch_size=batch_size,
                seed=seed,
                phase=PHASE_ORIGINAL,
                sample_indices=sample_indices,
                score_repeats=1,
                paired_seed=paired_seed,
            )
            for mode in modes:
                score_parts[mode].append(score_from_errors(errors, mode))
        else:
            for mode in modes:
                score_parts[mode].append(
                    score_samples(
                        model,
                        x_raw,
                        train_min,
                        train_max,
                        mode=mode,
                        n_masks=n_masks,
                        batch_size=batch_size,
                        seed=seed,
                        phase=PHASE_ORIGINAL,
                        sample_indices=sample_indices,
                        score_repeats=score_repeats,
                        paired_seed=paired_seed,
                    )
                )

        label_parts.append(chunk_labels)
        collected += len(positions)
        if max_samples is not None and collected >= max_samples:
            break

    scores = {
        mode: (
            np.concatenate(score_parts[mode])
            if score_parts[mode]
            else np.empty(0, dtype=np.float64)
        )
        for mode in modes
    }
    labels = (
        np.concatenate(label_parts)
        if label_parts
        else np.empty(0, dtype=np.str_)
    )
    return scores, labels, collected, dropped


def _thresholds(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return the two FPR thresholds calibrated on the BENIGN scores."""
    benign = scores[labels == BENIGN_LABEL]
    if benign.size == 0:
        raise ValueError("Cannot calibrate thresholds: no valid BENIGN rows")
    return {
        name: threshold_at_fpr(benign, fpr) for name, fpr in NORMAL_THRESHOLD_FPRS
    }


def run_validate(args: argparse.Namespace) -> int:
    """Run the dataset and model validation; return the process exit code."""
    data_dir, model_dir, meta_dir, _results_dir = _resolved_paths(args)
    train_csv = data_dir / "new_train.csv"
    test_csv = data_dir / "new_test.csv"

    for path in (train_csv, test_csv):
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

    train_min, train_max = load_metadata(meta_dir)
    model = load_model(model_dir)
    info = validate_dataset(train_csv, test_csv, meta_dir)
    benign = _benign_rows(train_csv)
    report = validate_model(model, train_min, train_max, benign, seed=args.seed)

    print(f"dataset: {_display_path(data_dir)}")
    print(
        f"train rows: {info.train_rows} | "
        f"test rows: {info.test_rows} "
        f"(valid: {info.test_valid_rows}, dropped: {info.test_dropped_invalid})"
    )
    counts = ", ".join(
        f"{label}={info.class_counts.get(label, 0)}" for label in info.class_counts
    )
    print(f"class counts: {counts}")
    print(f"constant features (max == min): {list(info.constant_features)}")
    print(
        f"model: input width {report['input_width']}, "
        f"benign error {report['benign_error']:.6g} < "
        f"noise error {report['noise_error']:.6g}"
    )
    print("validation ok")
    return 0


def run_normal(args: argparse.Namespace) -> int:
    """Run the normal evaluation; return the process exit code."""
    data_dir, model_dir, meta_dir, results_dir = _resolved_paths(args)
    test_csv = data_dir / "new_test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {test_csv}")

    modes = _parse_modes(args.modes)
    if "repo_plus" in modes and args.n_masks % REPO_PLUS_GROUPS != 0:
        raise ValueError(
            f"--n-masks must be divisible by {REPO_PLUS_GROUPS} for repo_plus; "
            f"got {args.n_masks}"
        )

    started_at = datetime.now(timezone.utc)
    train_min, train_max = load_metadata(meta_dir)
    model = load_model(model_dir)
    model_sha256 = _sha256_path(model_dir)
    test_sha256 = sha256_file(test_csv)

    scores_by_mode, labels, n_samples, n_dropped_invalid = _collect_scores(
        test_csv,
        model,
        train_min,
        train_max,
        modes=modes,
        n_masks=args.n_masks,
        batch_size=args.batch_size,
        seed=args.seed,
        score_repeats=args.score_repeats,
        paired_seed=args.seed if args.paired_scoring else None,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
    )
    finished_at = datetime.now(timezone.utc)
    runtime_seconds = (finished_at - started_at).total_seconds()
    if n_samples == 0:
        raise ValueError("No valid samples were read from the test CSV")

    environment = environment_manifest()
    commit = git_commit()
    dataset_block = {
        "name": "new_data",
        "test_csv": _display_path(test_csv),
        "test_csv_sha256": test_sha256,
        "n_samples": n_samples,
        "n_dropped_invalid": n_dropped_invalid,
    }
    scoring_block = {
        "n_masks": args.n_masks,
        "score_repeats": args.score_repeats,
        "paired_scoring": bool(args.paired_scoring),
    }
    sampling_block = {
        "mode": "full" if args.max_samples is None else "first_n",
        "max_samples": args.max_samples,
        "chunk_size": args.chunk_size,
        "n_samples": n_samples,
        "n_dropped_invalid": n_dropped_invalid,
    }

    output_dir = results_dir / "normal"
    for mode in modes:
        thresholds = _thresholds(scores_by_mode[mode], labels)
        config = {
            "seed": args.seed,
            "modes": list(modes),
            "thresholds": thresholds,
            "sampling": sampling_block,
            "scoring": dict(scoring_block),
            "model_sha256": model_sha256,
            "data_sha256": test_sha256,
            "code_version": __version__,
        }
        payload = build_normal_result(
            score_mode=mode,
            thresholds=thresholds,
            scores=scores_by_mode[mode],
            labels=labels,
            dataset=dataset_block,
            scoring=scoring_block,
            config=config,
            runtime_seconds=runtime_seconds,
            variant="primary",
            started_at=started_at,
            finished_at=finished_at,
            environment=environment,
            git_commit=commit,
        )
        path = output_dir / f"{mode}.json"
        write_json_atomic(path, payload)
        print(
            f"normal {mode}: {n_samples} samples, "
            f"thresholds {thresholds['fpr_0.01']:.6g}/{thresholds['fpr_0.10']:.6g}"
            f" -> {path}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the requested command."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse handles --help and usage errors itself
        code = exc.code
        return code if isinstance(code, int) else 1

    handlers = {
        "validate": run_validate,
        "normal": run_normal,
    }
    try:
        return int(handlers[args.command](args))
    except Exception as exc:  # noqa: BLE001 - keep the CLI exit code contract
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    raise SystemExit(main())
