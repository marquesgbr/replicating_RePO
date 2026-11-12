"""Single-pass distillation of the RePO+ ensemble (Task 4, Direction C).

The teacher is the 100-mask RePO+ score.  The student is a plain MLP that maps
the raw (normalized) flow to the teacher score in a single forward pass.  If a
single pass can reproduce the ranking and the TPR, the ensemble could be
dropped at inference; if it cannot, the negative result is the contribution.

The pilot measures the **normal** agreement only.  A definitive claim needs an
adaptive attack on the student, which is left to the full run (the attack
harness would need to optimize the student end-to-end).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mask_experiments.data import BENIGN_LABEL, FLOW_ATTACKS, load_flow_meta
from mask_experiments.runner import REPO_ROOT, load_flow_model
from mask_experiments.variance import load_sample, spearman
from new_data_eval.masking import POLICIES
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import score_samples, threshold_at_fpr

__all__ = ["run_distillation"]

DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 64
TRAIN_FRACTION = 0.7


def _macro_tpr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    tprs, counts = [], []
    for name in FLOW_ATTACKS:
        mask = labels == name
        if not mask.any():
            continue
        tprs.append(float(np.mean(scores[mask] >= threshold)))
        counts.append(int(mask.sum()))
    if not tprs:
        return {"macro": 0.0, "weighted": 0.0}
    tprs_array, counts_array = np.array(tprs), np.array(counts)
    return {
        "macro": float(tprs_array.mean()),
        "weighted": float(np.average(tprs_array, weights=counts_array)),
    }


def _split(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed + 3)
    train, test = [], []
    for name in sorted(set(labels)):
        positions = np.flatnonzero(labels == name)
        generator.shuffle(positions)
        cut = int(round(fraction * len(positions)))
        train.append(positions[:cut])
        test.append(positions[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def run_distillation(
    *,
    data_dir: Path,
    model_dir: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    n_masks: int = 100,
    seed: int = 2019,
    batch_size: int = 100,
    epochs: int = DEFAULT_EPOCHS,
    student_out: Path | None = None,
) -> dict[str, Any]:
    import tensorflow as tf

    started = time.perf_counter()
    x, labels, indices = load_sample(
        data_dir,
        cache=cache,
        max_benign=max_benign,
        max_per_attack=max_per_attack,
        seed=seed,
    )
    train_min, train_max = load_flow_meta(data_dir)
    teacher_model = load_flow_model(model_dir)
    policy = POLICIES["iid"]

    teacher_scores = score_samples(
        teacher_model,
        x,
        train_min,
        train_max,
        mode="repo_plus",
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        phase="original",
        sample_indices=indices,
        mask_policy=policy,
    )

    train_rows, test_rows = _split(labels, TRAIN_FRACTION, seed)
    x_norm = (x - train_min) / (train_max - train_min + 1e-6)
    x_train = x_norm[train_rows].astype(np.float32)
    x_test = x_norm[test_rows].astype(np.float32)
    y_train = np.log1p(teacher_scores[train_rows]).astype(np.float32)
    y_test = teacher_scores[test_rows]

    tf.keras.utils.set_random_seed(seed)
    student = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    student.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    history = student.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=DEFAULT_BATCH,
        verbose=0,
        shuffle=True,
    )
    predicted = np.expm1(student.predict(x_test, verbose=0).ravel())
    predicted = np.maximum(predicted, 0.0)

    test_labels = labels[test_rows]
    benign = test_labels == BENIGN_LABEL
    teacher_threshold = threshold_at_fpr(teacher_scores[test_rows][benign], 0.01)
    student_threshold = threshold_at_fpr(predicted[benign], 0.01)

    residuals = predicted - y_test
    student_path: str | None = None
    if student_out is not None:
        student_out = Path(student_out)
        student_out.parent.mkdir(parents=True, exist_ok=True)
        student.save(student_out)
        student_path = str(student_out)
    payload = {
        "schema_version": "mask_exp_v1",
        "experiment": "distillation",
        "dataset": {
            "name": "cicids2017-flow",
            "n_samples": int(len(labels)),
            "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows)),
            "seed": seed,
        },
        "teacher": {
            "n_masks": int(n_masks),
            "policy": policy.name,
            "threshold_fpr_0.01": float(teacher_threshold),
            "tpr": _macro_tpr(teacher_scores[test_rows], test_labels, teacher_threshold),
        },
        "student": {
            "architecture": "77-256-256-128-1",
            "epochs": int(epochs),
            "final_train_loss": float(history.history["loss"][-1]),
            "threshold_fpr_0.01": float(student_threshold),
            "tpr": _macro_tpr(predicted, test_labels, student_threshold),
            "spearman": spearman(predicted, y_test),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "mae": float(np.mean(np.abs(residuals))),
            "single_pass_forward": True,
            "adversarial_evaluated": False,
            "path": student_path,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    print(
        f"teacher tpr={payload['teacher']['tpr']['macro']:.3f} "
        f"student tpr={payload['student']['tpr']['macro']:.3f} "
        f"spearman={payload['student']['spearman']:.3f} "
        f"t={payload['runtime_seconds']:.0f}s"
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "flw_model"))
    parser.add_argument("--cache", default="")
    parser.add_argument("--max-benign", type=int, default=5_000)
    parser.add_argument("--max-per-attack", type=int, default=300)
    parser.add_argument("--n-masks", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--student-out",
        default=str(
            REPO_ROOT
            / "results"
            / "mask_efficiency"
            / "distillation_student.keras"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "mask_efficiency" / "distillation_flow.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_distillation(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        n_masks=args.n_masks,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        student_out=Path(args.student_out) if args.student_out else None,
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
