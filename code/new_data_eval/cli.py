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
import json
import logging
import os
import signal
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from new_data_eval import __version__
from new_data_eval.attack import FlowAttack
from new_data_eval.data import (
    ATTACK_LABELS,
    BENIGN_LABEL,
    DEFAULT_CHUNK_SIZE,
    N_FEATURES,
    canonical_label,
    integer_feature_indices,
    iter_test_chunks_detailed,
    load_metadata,
    select_attack_samples,
    sha256_file,
    validate_dataset,
)
from new_data_eval.results import (
    build_adversarial_result,
    build_normal_result,
    config_hash,
    environment_manifest,
    file_sha256,
    git_commit,
    write_json_atomic,
)
from new_data_eval.scoring import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_MASKS,
    PHASE_ORIGINAL,
    PHASE_VICTIM,
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
    "checkpoint_path",
    "load_normal_threshold",
    "main",
    "result_path",
    "run_adversarial",
    "run_normal",
    "run_smoke",
    "run_validate",
]

LOG = logging.getLogger(__name__)


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

#: Default number of samples of every attack (the paper uses the first 5000).
ADVERSARIAL_MAX_SAMPLES = 5000

#: Default checkpoint interval of the adversarial evaluation.
ADVERSARIAL_CHECKPOINT_EVERY = 50

#: Directory names of the adversarial variants.
ADVERSARIAL_VARIANTS: tuple[str, ...] = (
    "primary",
    "sensitivity_first_n",
    "sensitivity_paper_0.8",
)

#: Threshold policies of the adversarial evaluation.
THRESHOLD_POLICIES: tuple[str, ...] = ("calibrated", "paper_0.8")

#: Sample selection modes of the adversarial evaluation.
SAMPLE_MODES: tuple[str, ...] = ("random", "first")

#: Threshold policy and sample mode required by every non-smoke variant.
VARIANT_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "primary": ("calibrated", "random"),
    "sensitivity_first_n": ("calibrated", "first"),
    "sensitivity_paper_0.8": ("paper_0.8", "random"),
}

#: Benign rows scored to measure the FPR of the ``paper_0.8`` threshold.
BENIGN_FPR_ROWS = 5000

#: Sample budgets of the smoke command (normal / adversarial).
SMOKE_NORMAL_MAX_SAMPLES = 20
SMOKE_ADVERSARIAL_MAX_SAMPLES = 5

#: Mask count of the smoke command (RePO+ needs a multiple of five).
SMOKE_N_MASKS = 5

#: Exit code returned when an interruption asks the CLI to stop gracefully.
INTERRUPTED_EXIT_CODE = 143

#: Set to True by the SIGINT/SIGTERM handler; checked once per attack sample.
_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: object) -> None:
    """Mark the current run as interrupted without raising."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def stop_requested() -> bool:
    """Return True once an interruption signal has been received."""
    return _STOP_REQUESTED


@contextmanager
def _stop_signals() -> Iterator[None]:
    """Handle SIGINT/SIGTERM as a graceful stop and restore old handlers."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous: dict[int, Any] = {}
    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.getsignal(number)
            signal.signal(number, _request_stop)
        except (OSError, ValueError):
            continue
    try:
        yield
    finally:
        for number, handler in previous.items():
            try:
                signal.signal(number, handler)
            except (OSError, ValueError):
                pass


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

    adversarial = subparsers.add_parser(
        "adversarial", help="run the adversarial RePO+ evaluation"
    )
    _add_common_arguments(adversarial)
    _add_adversarial_arguments(adversarial)

    smoke = subparsers.add_parser(
        "smoke",
        help="run validate, normal and one adversarial attack under <results>/smoke",
    )
    _add_common_arguments(smoke)
    smoke.add_argument(
        "--attack",
        default="ddos",
        help="attack label of the smoke run (case-insensitive)",
    )
    smoke.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="masked rows sent to the model per forward pass",
    )

    return parser


def _add_adversarial_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the adversarial-only options to ``parser``."""
    parser.add_argument(
        "--attack",
        required=True,
        help="attack label (case-insensitive) or 'all' for the nine classes",
    )
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=ADVERSARIAL_MAX_SAMPLES,
        help="maximum samples drawn from every attack class",
    )
    parser.add_argument(
        "--sample-mode",
        choices=SAMPLE_MODES,
        default="random",
        help="random reservoir sampling or the first rows of the class",
    )
    parser.add_argument(
        "--n-masks",
        type=_positive_int,
        default=DEFAULT_N_MASKS,
        help="number of masks per sample (tests/smoke only)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="masked rows sent to the model per forward pass",
    )
    parser.add_argument(
        "--attack-threshold-policy",
        choices=THRESHOLD_POLICIES,
        default="calibrated",
        help="use the calibrated FPR 10%% threshold or 0.8 times it",
    )
    parser.add_argument(
        "--score-repeats",
        type=_positive_int,
        default=1,
        help="independent mask repeats whose scores are averaged",
    )
    parser.add_argument(
        "--paired-scoring",
        action="store_true",
        help="use the paired mask seed derivation",
    )
    parser.add_argument(
        "--variant",
        choices=ADVERSARIAL_VARIANTS,
        default="primary",
        help="result variant directory",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=ADVERSARIAL_CHECKPOINT_EVERY,
        help="samples between two checkpoints",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard an existing result, checkpoint and score file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned outputs without scoring anything",
    )


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
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int, int]:
    """Stream ``test_csv`` and return the scores, labels and row counts.

    The global row offset of every valid row is passed as ``sample_indices``,
    so the masks of a sample are the same whatever the chunk size is.  When
    ``max_samples`` is set, the reader stops after that many valid rows and the
    invalid rows after the last consumed row are not counted.

    Returns ``(scores_by_mode, labels, sample_indices, n_collected,
    n_dropped_invalid)``.
    """
    score_parts: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    label_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
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
        index_parts.append(sample_indices)
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
    sample_indices = (
        np.concatenate(index_parts)
        if index_parts
        else np.empty(0, dtype=np.int64)
    )
    return scores, labels, sample_indices, collected, dropped


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
    meta_sha256 = {
        "train_min": file_sha256(Path(meta_dir) / "train_min.npy"),
        "train_max": file_sha256(Path(meta_dir) / "train_max.npy"),
    }

    (
        scores_by_mode,
        labels,
        sample_indices,
        n_samples,
        n_dropped_invalid,
    ) = _collect_scores(
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
    scores_dir = results_dir / "scores"
    variant = str(getattr(args, "variant", "primary"))
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
            "meta_sha256": dict(meta_sha256),
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
            variant=variant,
            started_at=started_at,
            finished_at=finished_at,
            environment=environment,
            git_commit=commit,
        )
        path = output_dir / f"{mode}.json"
        write_json_atomic(path, payload)
        _write_npz_atomic(
            scores_dir / f"normal_{mode}.npz",
            {
                "scores": scores_by_mode[mode],
                "labels": labels,
                "sample_indices": sample_indices,
                "threshold_fpr_0.01": np.asarray(
                    [thresholds["fpr_0.01"]], dtype=np.float64
                ),
                "threshold_fpr_0.10": np.asarray(
                    [thresholds["fpr_0.10"]], dtype=np.float64
                ),
            },
        )
        print(
            f"normal {mode}: {n_samples} samples, "
            f"thresholds {thresholds['fpr_0.01']:.6g}/{thresholds['fpr_0.10']:.6g}"
            f" -> {path}"
        )
    return 0


# ---------------------------------------------------------------------------
# Adversarial evaluation
# ---------------------------------------------------------------------------


def _canonical_attack(attack: str) -> str:
    """Return the lowercase attack label or raise a useful error."""
    try:
        canonical = canonical_label(attack)
    except ValueError as exc:
        raise ValueError(
            f"{exc}; attacks must be one of {list(ATTACK_LABELS)} or 'all'"
        ) from exc
    if canonical not in ATTACK_LABELS:
        raise ValueError(
            f"Unknown attack label {canonical!r}; expected one of "
            f"{list(ATTACK_LABELS)} or 'all'"
        )
    return canonical


def _parse_attacks(value: str) -> list[str]:
    """Return the canonical attack labels requested by ``value``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("--attack must be a label or 'all'")
    if value.strip().lower() == "all":
        return list(ATTACK_LABELS)
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("--attack must be a label or 'all'")
    return list(dict.fromkeys(_canonical_attack(item) for item in labels))


def result_path(results_dir: Path, variant: str, attack: str) -> Path:
    """Return ``<results>/adversarial/<variant>/<attack>.json``."""
    canonical = _canonical_attack(attack)
    return Path(results_dir) / "adversarial" / str(variant) / f"{canonical}.json"


def checkpoint_path(results_dir: Path, variant: str, attack: str) -> Path:
    """Return the checkpoint path of one adversarial attack."""
    canonical = _canonical_attack(attack)
    return (
        Path(results_dir)
        / "adversarial"
        / str(variant)
        / f"{canonical}.checkpoint.npz"
    )


def _scores_path(results_dir: Path, variant: str, attack: str) -> Path:
    """Return the ``.npz`` score file of one adversarial attack."""
    canonical = _canonical_attack(attack)
    return Path(results_dir) / "scores" / f"{variant}_{canonical}.npz"


def _flat_result_path(results_dir: Path, attack: str) -> Path:
    """Return the smoke layout path (no variant directory)."""
    return Path(results_dir) / "adversarial" / f"{_canonical_attack(attack)}.json"


def _flat_checkpoint_path(results_dir: Path, attack: str) -> Path:
    """Return the smoke layout checkpoint path (no variant directory)."""
    return (
        Path(results_dir)
        / "adversarial"
        / f"{_canonical_attack(attack)}.checkpoint.npz"
    )


def _flat_scores_path(results_dir: Path, attack: str) -> Path:
    """Return the smoke layout score path (no variant prefix)."""
    return Path(results_dir) / "scores" / f"{_canonical_attack(attack)}.npz"


def load_normal_threshold(results_dir: Path) -> float:
    """Return ``thresholds.fpr_0.10`` of the normal RePO+ result."""
    path = Path(results_dir) / "normal" / "repo_plus.json"
    if not path.is_file():
        raise FileNotFoundError(f"Normal result not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(payload["thresholds"]["fpr_0.10"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Normal result {path} does not contain thresholds.fpr_0.10"
        ) from exc


def _read_normal_result(results_dir: Path) -> dict[str, Any] | None:
    """Return the normal RePO+ payload, or ``None`` when it is unreadable."""
    path = Path(results_dir) / "normal" / "repo_plus.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _benign_scores(
    test_csv: Path,
    model: Any,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    n_masks: int,
    batch_size: int,
    seed: int,
    score_repeats: int,
    paired_seed: int | None,
    chunk_size: int,
    max_rows: int | None,
) -> np.ndarray:
    """Score up to ``max_rows`` deterministic BENIGN rows of ``test_csv``.

    ``max_rows=None`` scores every valid BENIGN row.  The rows are taken in
    file order and the global row index is forwarded to ``score_samples``, so
    the scores do not depend on the chunk size.
    """
    parts: list[np.ndarray] = []
    collected = 0
    for chunk in iter_test_chunks_detailed(test_csv, chunk_size):
        positions = np.flatnonzero(
            chunk.valid & (chunk.labels == BENIGN_LABEL)
        )
        if positions.size == 0:
            continue
        if max_rows is not None:
            remaining = max_rows - collected
            if remaining <= 0:
                break
            positions = positions[:remaining]

        scores = score_samples(
            model,
            chunk.features[positions],
            train_min,
            train_max,
            mode="repo_plus",
            n_masks=n_masks,
            batch_size=batch_size,
            seed=seed,
            phase=PHASE_ORIGINAL,
            sample_indices=chunk.indices[positions],
            score_repeats=score_repeats,
            paired_seed=paired_seed,
        )
        parts.append(np.asarray(scores, dtype=np.float64))
        collected += len(positions)
        if max_rows is not None and collected >= max_rows:
            break

    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def _normal_result_path(results_dir: Path) -> Path:
    """Return the path of the normal RePO+ result read by the adversarial run."""
    return Path(results_dir) / "normal" / "repo_plus.json"


def _verify_normal_provenance(
    payload: Mapping[str, Any],
    *,
    path: Path,
    data_sha256: str,
    model_sha256: str,
    n_masks: int,
    score_repeats: int,
    paired_scoring: bool,
) -> None:
    """Raise when the stored normal result does not describe the current run.

    Reusing a threshold computed by another model, dataset or scoring setup
    would silently mix two experiments, so every provenance field that the
    normal result records is compared before the threshold is returned.
    """
    problems: list[str] = []

    if payload.get("model_type") != "new_data_flow":
        problems.append(
            "model_type is "
            f"{payload.get('model_type')!r}, expected 'new_data_flow'"
        )

    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        problems.append("missing dataset block")
        stored_data = None
    else:
        stored_data = dataset.get("test_csv_sha256")
    if stored_data != data_sha256:
        problems.append(
            "dataset.test_csv_sha256 is "
            f"{stored_data!r}, current test CSV is {data_sha256}"
        )

    config = payload.get("config")
    if not isinstance(config, Mapping):
        problems.append("missing config block")
        stored_model = None
    else:
        stored_model = config.get("model_sha256")
    if stored_model != model_sha256:
        problems.append(
            f"config.model_sha256 is {stored_model!r}, current model is "
            f"{model_sha256}"
        )

    scoring = payload.get("scoring")
    if not isinstance(scoring, Mapping):
        problems.append("missing scoring block")
        scoring = {}
    expected_scoring = {
        "n_masks": int(n_masks),
        "score_repeats": int(score_repeats),
        "paired_scoring": bool(paired_scoring),
    }
    for key, expected in expected_scoring.items():
        found = scoring.get(key)
        if found != expected:
            problems.append(
                f"scoring.{key} is {found!r}, current run uses {expected!r}"
            )

    if problems:
        raise ValueError(
            f"Normal result {path} does not match the current run: "
            + "; ".join(problems)
        )


def _resolve_real_threshold(
    results_dir: Path,
    test_csv: Path,
    model: Any,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    n_masks: int,
    batch_size: int,
    seed: int,
    score_repeats: int,
    paired_seed: int | None,
    chunk_size: int,
    model_sha256: str,
    data_sha256: str,
    paired_scoring: bool,
    force: bool,
) -> tuple[float, dict[str, Any] | None, dict[str, Any]]:
    """Return the FPR 10% threshold, the normal payload and its provenance.

    A stored normal result is only reused after its dataset, model, model type
    and scoring parameters match the current run.  ``--force`` bypasses the
    check explicitly and logs the mismatch.  When the result is missing (or
    unusable) the threshold is recomputed from the benign rows and the
    provenance block marks it as recomputed.
    """
    path = _normal_result_path(results_dir)
    normal_payload = _read_normal_result(results_dir)

    if normal_payload is not None:
        try:
            threshold = float(normal_payload["thresholds"]["fpr_0.10"])
            if not np.isfinite(threshold):
                raise ValueError("the stored threshold is not finite")
        except (KeyError, TypeError, ValueError) as exc:
            LOG.warning(
                "normal result %s is unusable (%s); recomputing the FPR 10%% "
                "threshold from %s",
                path,
                exc,
                test_csv,
            )
        else:
            try:
                _verify_normal_provenance(
                    normal_payload,
                    path=path,
                    data_sha256=data_sha256,
                    model_sha256=model_sha256,
                    n_masks=n_masks,
                    score_repeats=score_repeats,
                    paired_scoring=paired_scoring,
                )
            except ValueError as exc:
                if not force:
                    raise ValueError(
                        f"{exc}; rerun `normal` with the current model/data or "
                        "use --force to reuse the stored threshold"
                    ) from exc
                LOG.warning("%s; --force reuses the stored threshold", exc)
            return (
                threshold,
                normal_payload,
                {
                    "path": _display_path(path),
                    "config_hash": normal_payload.get("config_hash"),
                    "used": True,
                },
            )
    else:
        LOG.warning(
            "normal result %s is missing or invalid; recomputing the FPR 10%% "
            "threshold from %s",
            path,
            test_csv,
        )

    benign = _benign_scores(
        test_csv,
        model,
        train_min,
        train_max,
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        score_repeats=score_repeats,
        paired_seed=paired_seed,
        chunk_size=chunk_size,
        max_rows=None,
    )
    if benign.size == 0:
        raise ValueError(f"No valid BENIGN rows found in {test_csv}")
    return (
        threshold_at_fpr(benign, 0.10),
        None,
        {
            "path": _display_path(path),
            "config_hash": None,
            "used": False,
        },
    )


def _fpr_at_attack_threshold(
    policy: str,
    real_threshold: float,
    attack_threshold: float,
    normal_payload: dict[str, Any] | None,
    test_csv: Path,
    model: Any,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    n_masks: int,
    batch_size: int,
    seed: int,
    score_repeats: int,
    paired_seed: int | None,
    chunk_size: int,
) -> float:
    """Return the FPR measured on benign rows at ``attack_threshold``.

    The calibrated policy reuses the FPR that the normal evaluation measured
    at its FPR 10% threshold.  When that result is unavailable, or the policy
    moves the threshold (``paper_0.8``), the benign rows are scored again and
    their measured FPR at ``attack_threshold`` is returned -- the value is
    never fabricated from the nominal calibration target.
    """
    if policy == "calibrated" and normal_payload is not None:
        try:
            return float(normal_payload["fpr"]["at_fpr_0.10_threshold"])
        except (KeyError, TypeError, ValueError):
            LOG.warning(
                "normal result has no measured FPR; measuring it from %s",
                test_csv,
            )

    benign = _benign_scores(
        test_csv,
        model,
        train_min,
        train_max,
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        score_repeats=score_repeats,
        paired_seed=paired_seed,
        chunk_size=chunk_size,
        max_rows=BENIGN_FPR_ROWS,
    )
    if benign.size == 0:
        raise ValueError(f"No valid BENIGN rows found in {test_csv}")
    return float(np.mean(benign >= attack_threshold))


def _stable_run_config(
    *,
    attack: str,
    variant: str,
    seed: int,
    sample_mode: str,
    max_samples: int,
    n_masks: int,
    batch_size: int,
    score_repeats: int,
    paired_scoring: bool,
    policy: str,
    real_threshold: float,
    attack_threshold: float,
    model_sha256: str,
    data_sha256: str,
    meta_sha256: Mapping[str, str],
    normal_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable parameters hashed into the checkpoint.

    The adversarial ``config_hash`` of the result builder also covers the
    mutable per-sample counts, so it cannot be used to decide whether a
    checkpoint belongs to the current run.  This mapping deliberately holds
    only parameters that must not change while a checkpoint is resumed.
    """
    return {
        "attack": attack,
        "variant": variant,
        "seed": int(seed),
        "sample_mode": sample_mode,
        "max_samples": int(max_samples),
        "n_masks": int(n_masks),
        "batch_size": int(batch_size),
        "score_repeats": int(score_repeats),
        "paired_scoring": bool(paired_scoring),
        "threshold_policy": policy,
        "real_threshold": float(real_threshold),
        "attack_threshold": float(attack_threshold),
        "model_sha256": model_sha256,
        "data_sha256": data_sha256,
        "meta_sha256": dict(meta_sha256),
        "normal_result": dict(normal_result),
        "code_version": __version__,
    }


def _empty_counts() -> dict[str, int]:
    """Return the count block of a fresh adversarial run."""
    return {
        "n": 0,
        "fooled": 0,
        "evaded": 0,
        "backup": 0,
        "cons_as_mal": 0,
        "cons_as_ben": 0,
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed ``.npz`` file atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".npz", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_arrays(
    attack: FlowAttack,
    stable_hash: str,
    *,
    next_index: int,
    original_scores: np.ndarray,
    victim_scores: np.ndarray,
    x_adv: np.ndarray,
    kinds: np.ndarray,
    counts: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """Return the arrays stored in a checkpoint."""
    arrays: dict[str, np.ndarray] = {
        "config_hash": np.asarray([stable_hash], dtype=np.str_),
        "next_index": np.asarray(next_index, dtype=np.int64),
        "n_samples": np.asarray(len(original_scores), dtype=np.int64),
        "original_scores": np.asarray(original_scores, dtype=np.float64),
        "victim_scores": np.asarray(victim_scores, dtype=np.float64),
        "x_adv": np.asarray(x_adv, dtype=np.float32),
        "kinds": np.asarray(kinds, dtype=np.str_),
    }
    for name, value in counts.items():
        arrays[f"count_{name}"] = np.asarray(int(value), dtype=np.int64)
    arrays.update(attack.optimizer_state())
    return arrays


def _load_checkpoint(
    path: Path, stable_hash: str, n_samples: int
) -> dict[str, Any] | None:
    """Load a checkpoint whose immutable configuration matches."""
    path = Path(path)
    if not path.is_file():
        return None

    with np.load(path, allow_pickle=False) as data:
        saved_hash = ""
        if "config_hash" in data.files:
            saved_hash = str(np.asarray(data["config_hash"]).ravel()[0])
        if saved_hash != stable_hash:
            raise RuntimeError(
                "checkpoint found with a different configuration; use --force "
                f"to discard it ({path})"
            )
        required = {
            "next_index",
            "original_scores",
            "victim_scores",
            "x_adv",
            "kinds",
        }
        if not required.issubset(data.files):
            raise RuntimeError(
                "checkpoint is incomplete; use --force to discard it "
                f"({path})"
            )

        original_scores = np.array(data["original_scores"], dtype=np.float64)
        victim_scores = np.array(data["victim_scores"], dtype=np.float64)
        x_adv = np.array(data["x_adv"], dtype=np.float32)
        kinds = np.array(data["kinds"], dtype=np.str_)
        next_index = int(np.asarray(data["next_index"]).ravel()[0])
        if (
            original_scores.shape != (n_samples,)
            or victim_scores.shape != (n_samples,)
            or x_adv.shape != (n_samples, N_FEATURES)
            or kinds.shape != (n_samples,)
            or not 0 <= next_index <= n_samples
        ):
            raise RuntimeError(
                "checkpoint arrays do not match the current sampling; use "
                f"--force to discard it ({path})"
            )
        counts = {
            name: int(np.asarray(data[f"count_{name}"]).ravel()[0])
            for name in _empty_counts()
        }
        optimizer_state = {
            name: np.array(data[name], copy=True)
            for name in data.files
            if name.startswith("flow_opt_")
        }

    return {
        "next_index": next_index,
        "original_scores": original_scores,
        "victim_scores": victim_scores,
        "x_adv": x_adv,
        "kinds": kinds,
        "counts": counts,
        "optimizer_state": optimizer_state,
    }


def _remove_if_exists(*paths: Path) -> None:
    """Delete every existing path (``--force`` semantics)."""
    for path in paths:
        if Path(path).exists():
            Path(path).unlink()


def _run_attack(
    attack: str,
    *,
    test_csv: Path,
    model: Any,
    train_min: np.ndarray,
    train_max: np.ndarray,
    model_sha256: str,
    data_sha256: str,
    meta_sha256: Mapping[str, str],
    normal_result: Mapping[str, Any],
    real_threshold: float,
    attack_threshold: float,
    fpr_at_attack: float,
    decimal_features: Sequence[int],
    environment: Mapping[str, Any],
    commit: str | None,
    args: argparse.Namespace,
    flat: bool,
) -> int:
    """Run one attack class and write its checkpoint, scores and result."""
    started_at = datetime.now(timezone.utc)
    x_mal, n_available = select_attack_samples(
        test_csv,
        attack,
        args.max_samples,
        args.sample_mode,
        args.seed,
        chunksize=args.chunk_size,
    )
    n_samples = int(len(x_mal))
    if n_samples == 0:
        raise ValueError(f"No samples available for attack {attack!r}")

    variant = str(getattr(args, "variant", "smoke"))
    if flat:
        output_result = _flat_result_path(Path(args.results_dir), attack)
        output_checkpoint = _flat_checkpoint_path(Path(args.results_dir), attack)
        output_scores = _flat_scores_path(Path(args.results_dir), attack)
    else:
        output_result = result_path(Path(args.results_dir), variant, attack)
        output_checkpoint = checkpoint_path(
            Path(args.results_dir), variant, attack
        )
        output_scores = _scores_path(Path(args.results_dir), variant, attack)

    if args.force:
        _remove_if_exists(output_result, output_checkpoint, output_scores)

    paired_seed = args.seed if args.paired_scoring else None
    stable_config = _stable_run_config(
        attack=attack,
        variant=variant,
        seed=args.seed,
        sample_mode=args.sample_mode,
        max_samples=args.max_samples,
        n_masks=args.n_masks,
        batch_size=args.batch_size,
        score_repeats=args.score_repeats,
        paired_scoring=bool(args.paired_scoring),
        policy=args.attack_threshold_policy,
        real_threshold=real_threshold,
        attack_threshold=attack_threshold,
        model_sha256=model_sha256,
        data_sha256=data_sha256,
        meta_sha256=meta_sha256,
        normal_result=normal_result,
    )
    stable_hash = config_hash(stable_config)

    attack_runner = FlowAttack(
        model,
        train_min,
        train_max,
        decimal_features,
        N_FEATURES,
        real_threshold,
        attack_threshold,
        seed=args.seed,
        n_masks=args.n_masks,
    )

    checkpoint = _load_checkpoint(output_checkpoint, stable_hash, n_samples)
    if checkpoint is None:
        original_scores = score_samples(
            model,
            x_mal,
            train_min,
            train_max,
            mode="repo_plus",
            n_masks=args.n_masks,
            batch_size=args.batch_size,
            seed=args.seed,
            phase=PHASE_ORIGINAL,
            sample_indices=np.arange(n_samples, dtype=np.int64),
            score_repeats=args.score_repeats,
            paired_seed=paired_seed,
        )
        original_scores = np.asarray(original_scores, dtype=np.float64)
        victim_scores = np.zeros(n_samples, dtype=np.float64)
        x_adv = np.zeros((n_samples, N_FEATURES), dtype=np.float32)
        kinds = np.full(n_samples, "pending", dtype="U12")
        counts = _empty_counts()
        next_index = 0
    else:
        original_scores = checkpoint["original_scores"]
        victim_scores = checkpoint["victim_scores"]
        x_adv = checkpoint["x_adv"]
        kinds = checkpoint["kinds"]
        counts = checkpoint["counts"]
        next_index = checkpoint["next_index"]
        attack_runner.load_optimizer_state(checkpoint["optimizer_state"])

    def save_checkpoint(index: int) -> None:
        _write_npz_atomic(
            output_checkpoint,
            _checkpoint_arrays(
                attack_runner,
                stable_hash,
                next_index=index,
                original_scores=original_scores,
                victim_scores=victim_scores,
                x_adv=x_adv,
                kinds=kinds,
                counts=counts,
            ),
        )

    for index in range(next_index, n_samples):
        outcome = attack_runner.find_adv(x_mal[index], sample_index=index)
        kind = outcome.kind
        if kind == "no_change":
            counts["cons_as_ben"] += 1
            victim_sample = x_mal[index]
        elif kind == "not_fooled":
            counts["cons_as_mal"] += 1
            victim_sample = x_mal[index]
        elif kind == "evaded":
            counts["fooled"] += 1
            counts["evaded"] += 1
            victim_sample = outcome.x_adv
        elif kind == "backup":
            counts["fooled"] += 1
            counts["backup"] += 1
            victim_sample = outcome.x_adv
        else:  # pragma: no cover - AttackOutcome validates its kind
            raise ValueError(f"Unknown attack outcome: {kind!r}")

        if victim_sample is None:  # pragma: no cover - defensive
            victim_sample = x_mal[index]
        x_adv[index] = np.asarray(victim_sample, dtype=np.float32)
        kinds[index] = kind

        checkpoint_due = (index + 1) % args.checkpoint_every == 0
        interrupted = stop_requested()
        if checkpoint_due or interrupted:
            save_checkpoint(index + 1)
            LOG.info("checkpoint %s: %d/%d", attack, index + 1, n_samples)
        if interrupted:
            LOG.warning(
                "interrupted %s after %d/%d samples; checkpoint saved",
                attack,
                index + 1,
                n_samples,
            )
            return INTERRUPTED_EXIT_CODE

    counts["n"] = n_samples
    # The victim scores are computed once, with the local sample positions as
    # the stateless mask indices, so --score-repeats is averaged by
    # score_samples exactly like the original scores above.
    victim_scores = np.asarray(
        score_samples(
            model,
            x_adv,
            train_min,
            train_max,
            mode="repo_plus",
            n_masks=args.n_masks,
            batch_size=args.batch_size,
            seed=args.seed,
            phase=PHASE_VICTIM,
            sample_indices=np.arange(n_samples, dtype=np.int64),
            score_repeats=args.score_repeats,
            paired_seed=paired_seed,
        ),
        dtype=np.float64,
    )
    save_checkpoint(n_samples)

    normal_tpr = (
        float(np.mean(original_scores >= attack_threshold))
        if n_samples
        else 0.0
    )
    victim_tpr = (
        float(np.mean(victim_scores >= attack_threshold))
        if n_samples
        else 0.0
    )
    _write_npz_atomic(
        output_scores,
        {
            "sample_indices": np.arange(n_samples, dtype=np.int64),
            "original_scores": original_scores,
            "victim_scores": victim_scores,
            "kinds": kinds,
            "real_threshold": np.asarray([real_threshold], dtype=np.float64),
            "attack_threshold": np.asarray([attack_threshold], dtype=np.float64),
        },
    )

    finished_at = datetime.now(timezone.utc)
    payload = build_adversarial_result(
        attack_type=attack,
        variant=variant,
        score_mode="repo_plus",
        dataset={
            "n_available": int(n_available),
            "n_samples": n_samples,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
        },
        thresholds={
            "fpr_0.10": real_threshold,
            "attack": attack_threshold,
            "policy": args.attack_threshold_policy,
        },
        scoring={
            "n_masks": args.n_masks,
            "score_repeats": args.score_repeats,
            "paired_scoring": bool(args.paired_scoring),
        },
        counts=counts,
        fpr_at_attack_threshold=fpr_at_attack,
        normal_tpr_at_attack_threshold=normal_tpr,
        victim_tpr=victim_tpr,
        config=stable_config,
        runtime_seconds=(finished_at - started_at).total_seconds(),
        started_at=started_at,
        finished_at=finished_at,
        environment=environment,
        git_commit=commit,
    )
    write_json_atomic(output_result, payload)
    print(
        f"adversarial {attack}: {n_samples}/{n_available} samples, "
        f"fooled={counts['fooled']} (evaded={counts['evaded']}, "
        f"backup={counts['backup']}), victim_tpr={victim_tpr:.6g} "
        f"-> {output_result}"
    )
    return 0


def _validate_variant_combination(
    variant: str, policy: str, sample_mode: str
) -> None:
    """Reject a variant whose policy or sampling mode does not belong to it.

    Each non-smoke variant represents one analysis: the main result uses a
    random sample and the calibrated threshold, ``first_n`` keeps the prefix
    and the calibrated threshold, and ``paper_0.8`` uses a random sample with
    the sensitivity threshold.  Mixing them would overwrite or mislabel a
    result, so the combination is validated before anything is written.
    """
    required = VARIANT_REQUIREMENTS.get(variant)
    if required is None:
        return
    expected_policy, expected_mode = required
    if policy != expected_policy or sample_mode != expected_mode:
        raise ValueError(
            f"variant {variant!r} requires threshold policy "
            f"{expected_policy!r} and sample mode {expected_mode!r}; got "
            f"{policy!r} and {sample_mode!r}"
        )


def _run_adversarial(args: argparse.Namespace, *, flat: bool) -> int:
    """Shared implementation of ``adversarial`` and the smoke attack."""
    data_dir, model_dir, meta_dir, results_dir = _resolved_paths(args)
    test_csv = data_dir / "new_test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {test_csv}")

    attacks = _parse_attacks(args.attack)
    if args.n_masks % REPO_PLUS_GROUPS != 0:
        raise ValueError(
            f"--n-masks must be divisible by {REPO_PLUS_GROUPS} for repo_plus; "
            f"got {args.n_masks}"
        )
    if args.sample_mode not in SAMPLE_MODES:
        raise ValueError(
            f"Unknown sample mode {args.sample_mode!r}; expected one of "
            f"{list(SAMPLE_MODES)}"
        )
    if not flat and args.variant not in ADVERSARIAL_VARIANTS:
        raise ValueError(
            f"Unknown variant {args.variant!r}; expected one of "
            f"{list(ADVERSARIAL_VARIANTS)}"
        )
    if args.attack_threshold_policy not in THRESHOLD_POLICIES:
        raise ValueError(
            f"Unknown threshold policy {args.attack_threshold_policy!r}; "
            f"expected one of {list(THRESHOLD_POLICIES)}"
        )
    if not flat:
        _validate_variant_combination(
            args.variant, args.attack_threshold_policy, args.sample_mode
        )

    if getattr(args, "dry_run", False):
        for attack in attacks:
            if flat:
                destination = _flat_result_path(results_dir, attack)
            else:
                destination = result_path(results_dir, args.variant, attack)
            print(f"dry run: {attack} -> {destination}")
        return 0

    train_min, train_max = load_metadata(meta_dir)
    model = load_model(model_dir)
    model_sha256 = _sha256_path(model_dir)
    data_sha256 = sha256_file(test_csv)
    meta_sha256 = {
        "train_min": file_sha256(meta_dir / "train_min.npy"),
        "train_max": file_sha256(meta_dir / "train_max.npy"),
    }

    # The reference derives this list once from the concatenated valid test
    # matrix, not from the selected attack samples, so every attack shares it.
    decimal_features = integer_feature_indices(
        test_csv, chunksize=args.chunk_size
    )

    paired_seed = args.seed if args.paired_scoring else None
    (
        real_threshold,
        normal_payload,
        normal_result,
    ) = _resolve_real_threshold(
        results_dir,
        test_csv,
        model,
        train_min,
        train_max,
        n_masks=args.n_masks,
        batch_size=args.batch_size,
        seed=args.seed,
        score_repeats=args.score_repeats,
        paired_seed=paired_seed,
        chunk_size=args.chunk_size,
        model_sha256=model_sha256,
        data_sha256=data_sha256,
        paired_scoring=bool(args.paired_scoring),
        force=bool(args.force),
    )
    attack_threshold = (
        real_threshold
        if args.attack_threshold_policy == "calibrated"
        else 0.8 * real_threshold
    )
    fpr_at_attack = _fpr_at_attack_threshold(
        args.attack_threshold_policy,
        real_threshold,
        attack_threshold,
        normal_payload,
        test_csv,
        model,
        train_min,
        train_max,
        n_masks=args.n_masks,
        batch_size=args.batch_size,
        seed=args.seed,
        score_repeats=args.score_repeats,
        paired_seed=paired_seed,
        chunk_size=args.chunk_size,
    )
    environment = environment_manifest()
    commit = git_commit()

    for attack in attacks:
        exit_code = _run_attack(
            attack,
            test_csv=test_csv,
            model=model,
            train_min=train_min,
            train_max=train_max,
            model_sha256=model_sha256,
            data_sha256=data_sha256,
            meta_sha256=meta_sha256,
            normal_result=normal_result,
            real_threshold=real_threshold,
            attack_threshold=attack_threshold,
            fpr_at_attack=fpr_at_attack,
            decimal_features=decimal_features,
            environment=environment,
            commit=commit,
            args=args,
            flat=flat,
        )
        if exit_code != 0:
            return exit_code
    return 0


def run_adversarial(args: argparse.Namespace) -> int:
    """Run the adversarial evaluation; return the process exit code."""
    with _stop_signals():
        return _run_adversarial(args, flat=False)


def run_smoke(args: argparse.Namespace) -> int:
    """Run the isolated smoke pipeline under ``<results-dir>/smoke``."""
    _data_dir, _model_dir, _meta_dir, results_dir = _resolved_paths(args)
    smoke_dir = results_dir / "smoke"

    run_validate(args)

    normal_args = argparse.Namespace(**vars(args))
    normal_args.results_dir = str(smoke_dir)
    normal_args.modes = "repo,repo_plus"
    normal_args.max_samples = SMOKE_NORMAL_MAX_SAMPLES
    normal_args.n_masks = SMOKE_N_MASKS
    normal_args.score_repeats = 1
    normal_args.paired_scoring = False
    normal_args.variant = "smoke"
    run_normal(normal_args)

    adversarial_args = argparse.Namespace(**vars(args))
    adversarial_args.results_dir = str(smoke_dir)
    adversarial_args.attack = args.attack
    adversarial_args.max_samples = SMOKE_ADVERSARIAL_MAX_SAMPLES
    adversarial_args.n_masks = SMOKE_N_MASKS
    adversarial_args.sample_mode = "random"
    adversarial_args.attack_threshold_policy = "calibrated"
    adversarial_args.score_repeats = 1
    adversarial_args.paired_scoring = False
    adversarial_args.variant = "smoke"
    adversarial_args.checkpoint_every = ADVERSARIAL_CHECKPOINT_EVERY
    adversarial_args.force = False
    adversarial_args.dry_run = False
    with _stop_signals():
        return _run_adversarial(adversarial_args, flat=True)


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
        "adversarial": run_adversarial,
        "smoke": run_smoke,
    }
    try:
        return int(handlers[args.command](args))
    except Exception as exc:  # noqa: BLE001 - keep the CLI exit code contract
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    raise SystemExit(main())
