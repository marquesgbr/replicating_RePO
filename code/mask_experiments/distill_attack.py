"""Adversarial evaluation of the distilled student (Direction C).

The student predicts the teacher's RePO+ score in a single forward pass.  This
module attacks the **student directly** (white-box, same feasibility projection
as the reference flow attack) and measures, on the same adversarial samples:

* the student's victim TPR at its own FPR-10 % threshold;
* the teacher's victim TPR at the teacher threshold (paired comparison);
* the attack cost (forward passes / masked rows).

A student that keeps the teacher's adversarial TPR would make the ensemble
dispensable; a student whose TPR collapses shows that the ensemble is the
defense.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mask_experiments.data import BENIGN_LABEL, load_flow_meta
from mask_experiments.runner import REPO_ROOT, load_flow_model
from mask_experiments.variance import _macro_tpr, load_sample
from new_data_eval.attack import FlowAttack
from new_data_eval.masking import POLICIES
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import (
    PHASE_ORIGINAL,
    PHASE_VICTIM,
    ScoringCost,
    score_samples,
    threshold_at_fpr,
)

__all__ = ["DEFAULT_ATTACKS", "StudentAttack", "run_distill_attack"]

DEFAULT_ATTACKS: tuple[str, ...] = (
    "FTP-Patator",
    "DoS slowloris",
    "DoS Hulk",
    "DDoS",
)

ATTACK_FACTOR = 0.8


def _tensorflow():
    import tensorflow as tensorflow

    return tensorflow


class StudentAttack(FlowAttack):
    """Reference flow attack whose score and loss are the student's output."""

    def __init__(self, student: Any, *args: Any, **kwargs: Any) -> None:
        self.student = student
        super().__init__(student, *args, **kwargs)

    def _make_optim(self, t):
        @t.function
        def optim(x, adv_mask, def_mask, optimizer):
            del def_mask  # the student is deterministic: no random mask
            with t.GradientTape() as tape:
                span = self.train_max - self.train_min + 1e-6
                x_normalized = (x - self.train_min) / span
                adv = x_normalized + self.alpha * adv_mask
                predicted = self.student(adv, training=False)
                loss = t.reduce_sum(predicted)
            gradients = tape.gradient(loss, [self.alpha])
            optimizer.apply_gradients(zip(gradients, [self.alpha]))

        return optim

    def score(self, sample: np.ndarray, sample_index: int, phase: str) -> float:
        del sample_index, phase
        batch = self._as_sample(sample)[None, :]
        span = self.train_max - self.train_min + 1e-6
        x_normalized = (batch - self.train_min) / span
        predicted = self.student(x_normalized, training=False)
        value = float(np.expm1(np.asarray(predicted).ravel()[0]))
        return max(0.0, value)


def _student_scores(student: Any, x_raw: np.ndarray, train_min, train_max) -> np.ndarray:
    t = _tensorflow()
    span = train_max - train_min + 1e-6
    x_normalized = (np.asarray(x_raw, dtype=np.float32) - train_min) / span
    predicted = student(x_normalized, training=False)
    return np.maximum(0.0, np.expm1(np.asarray(predicted).ravel()).astype(np.float64))


def run_distill_attack(
    *,
    data_dir: Path,
    model_dir: Path,
    student_path: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    per_attack: int = 20,
    attacks: Iterable[str] = DEFAULT_ATTACKS,
    seed: int = 2019,
    batch_size: int = 100,
) -> dict[str, Any]:
    t = _tensorflow()
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
    student = t.keras.models.load_model(str(student_path))
    policy = POLICIES["iid"]
    benign = labels == BENIGN_LABEL
    generator = np.random.default_rng(seed + 5)
    decimal = [
        index
        for index in range(x.shape[1])
        if np.sum(np.abs(x[:, index] - np.round(x[:, index]))) == 0
    ]

    cost = ScoringCost()
    student_benign = _student_scores(student, x[benign], train_min, train_max)
    teacher_benign = score_samples(
        teacher_model,
        x[benign],
        train_min,
        train_max,
        mode="repo_plus",
        n_masks=100,
        batch_size=batch_size,
        seed=seed,
        phase=PHASE_ORIGINAL,
        sample_indices=indices[benign],
        mask_policy=policy,
        cost=cost,
    )
    student_threshold = threshold_at_fpr(student_benign, 0.10)
    teacher_threshold = threshold_at_fpr(teacher_benign, 0.10)
    student_attack = ATTACK_FACTOR * student_threshold
    teacher_attack = ATTACK_FACTOR * teacher_threshold

    selected: dict[str, np.ndarray] = {}
    for attack in attacks:
        positions = np.flatnonzero(labels == attack)
        take = min(per_attack, len(positions))
        if take:
            selected[attack] = np.sort(
                generator.choice(positions, size=take, replace=False)
            )

    payload: dict[str, Any] = {
        "schema_version": "mask_exp_v1",
        "experiment": "distill_attack",
        "dataset": {
            "name": "cicids2017-flow",
            "n_benign": int(benign.sum()),
            "attacks": {name: int(len(rows)) for name, rows in selected.items()},
            "seed": seed,
            "attack_factor": ATTACK_FACTOR,
        },
        "thresholds": {
            "student": {
                "fpr_0.10": float(student_threshold),
                "attack": float(student_attack),
            },
            "teacher": {
                "fpr_0.10": float(teacher_threshold),
                "attack": float(teacher_attack),
            },
        },
        "attacks": {},
        "runtime_seconds": 0.0,
    }

    for attack_name, rows in selected.items():
        runner = StudentAttack(
            student,
            train_min,
            train_max,
            decimal,
            x.shape[1],
            student_threshold,
            student_attack,
            seed=seed,
            n_masks=100,
            mask_policy=None,
        )
        outcomes = []
        victims = []
        for position, sample_index in zip(rows, indices[rows]):
            outcome = runner.find_adv(x[position], int(sample_index))
            victims.append(
                x[position]
                if outcome.x_adv is None
                else np.asarray(outcome.x_adv, dtype=np.float32)
            )
            outcomes.append(outcome)

        victim_array = np.stack(victims)
        student_victim = _student_scores(
            student, victim_array, train_min, train_max
        )
        student_original = _student_scores(student, x[rows], train_min, train_max)
        teacher_victim = score_samples(
            teacher_model,
            victim_array,
            train_min,
            train_max,
            mode="repo_plus",
            n_masks=100,
            batch_size=batch_size,
            seed=seed,
            phase=PHASE_VICTIM,
            sample_indices=indices[rows],
            mask_policy=policy,
            cost=cost,
        )
        teacher_original = score_samples(
            teacher_model,
            x[rows],
            train_min,
            train_max,
            mode="repo_plus",
            n_masks=100,
            batch_size=batch_size,
            seed=seed,
            phase=PHASE_ORIGINAL,
            sample_indices=indices[rows],
            mask_policy=policy,
            cost=cost,
        )
        student_normal = student_original >= student_attack
        teacher_normal = teacher_original >= teacher_attack
        student_victim_tpr = float(np.mean(student_victim >= student_attack))
        teacher_victim_tpr = float(np.mean(teacher_victim >= teacher_attack))
        payload["attacks"][attack_name] = {
            "n": int(len(rows)),
            "student_normal_tpr": float(student_normal.mean()),
            "student_victim_tpr": student_victim_tpr,
            "teacher_normal_tpr": float(teacher_normal.mean()),
            "teacher_victim_tpr": teacher_victim_tpr,
            "student_fooled": int(np.sum(student_normal & ~(student_victim >= student_attack))),
            "teacher_fooled": int(np.sum(teacher_normal & ~(teacher_victim >= teacher_attack))),
            "kinds": {
                kind: int(sum(outcome.kind == kind for outcome in outcomes))
                for kind in ("evaded", "backup", "no_change", "not_fooled")
            },
            "masks_evaluated": int(
                np.mean([outcome.masks_evaluated for outcome in outcomes])
            ),
        }
        print(
            f"distill-attack {attack_name}: student={student_victim_tpr:.2f} "
            f"teacher={teacher_victim_tpr:.2f} "
            f"fooled={payload['attacks'][attack_name]['student_fooled']}/{len(rows)}"
        )

    payload["runtime_seconds"] = time.perf_counter() - started
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "flw_model"))
    parser.add_argument(
        "--student",
        default=str(
            REPO_ROOT
            / "results"
            / "mask_efficiency"
            / "distillation_student.keras"
        ),
    )
    parser.add_argument("--cache", default="")
    parser.add_argument("--max-benign", type=int, default=5_000)
    parser.add_argument("--max-per-attack", type=int, default=300)
    parser.add_argument("--per-attack", type=int, default=20)
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "mask_efficiency" / "distill_attack.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_distill_attack(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        student_path=Path(args.student),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        per_attack=args.per_attack,
        attacks=[item.strip() for item in args.attacks.split(",") if item.strip()],
        seed=args.seed,
        batch_size=args.batch_size,
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
