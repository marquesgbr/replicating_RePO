"""Adversarial pilot of the mask policies (Task 2, Direction A).

Every variant is attacked by an adversary that **knows the detector**: the
optimization masks, the number of masks and the RePO+ success/victim scores all
use the same mask policy as the detector.  The threshold of the variant is
recalibrated on its own benign scores at FPR = 10 % (and the flow convention of
``attack_threshold = 0.8 * real_threshold`` is kept), so the comparison is at a
matched false positive rate.

The pilot is deliberately small (a few dozen samples per attack) because the
attack is the expensive step; the same harness runs the full arrays unchanged.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from mask_experiments.data import BENIGN_LABEL, load_flow_meta
from mask_experiments.runner import REPO_ROOT, load_flow_model
from mask_experiments.variance import load_sample
from new_data_eval.attack import FlowAttack
from new_data_eval.masking import POLICIES, MaskPolicy
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import (
    PHASE_ATTACK,
    PHASE_ORIGINAL,
    PHASE_VICTIM,
    ScoringCost,
    score_samples,
    threshold_at_fpr,
)

__all__ = ["DEFAULT_ATTACKS", "DEFAULT_VARIANTS", "run_adversarial"]

DEFAULT_ATTACKS: tuple[str, ...] = (
    "FTP-Patator",
    "DoS slowloris",
    "DoS Hulk",
    "DDoS",
)

#: ``(policy, n_masks)`` pairs of the pilot.
DEFAULT_VARIANTS: tuple[tuple[str, int], ...] = (
    ("iid", 100),
    ("iid", 25),
    ("balanced", 25),
    ("covering", 25),
)

DEFAULT_FPR = 0.10
ATTACK_FACTOR = 0.8


def _decimal_features(x: np.ndarray) -> list[int]:
    return [
        index
        for index in range(x.shape[1])
        if np.sum(np.abs(x[:, index] - np.round(x[:, index]))) == 0
    ]


def _thresholds(
    model: Any,
    benign: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    policy: MaskPolicy,
    n_masks: int,
    seed: int,
    batch_size: int,
) -> tuple[float, float]:
    scores = score_samples(
        model,
        benign,
        train_min,
        train_max,
        mode="repo_plus",
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        phase=PHASE_ORIGINAL,
        score_repeats=1,
        mask_policy=policy,
    )
    real = threshold_at_fpr(scores, DEFAULT_FPR)
    return real, ATTACK_FACTOR * real


def run_adversarial(
    *,
    data_dir: Path,
    model_dir: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    per_attack: int = 20,
    attacks: Iterable[str] = DEFAULT_ATTACKS,
    variants: Iterable[tuple[str, int]] = DEFAULT_VARIANTS,
    seed: int = 2019,
    batch_size: int = 100,
) -> dict[str, Any]:
    started = time.perf_counter()
    x, labels, indices = load_sample(
        data_dir,
        cache=cache,
        max_benign=max_benign,
        max_per_attack=max_per_attack,
        seed=seed,
    )
    train_min, train_max = load_flow_meta(data_dir)
    model = load_flow_model(model_dir)
    decimal = _decimal_features(x)
    generator = np.random.default_rng(seed + 2)

    benign = x[labels == BENIGN_LABEL]
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
        "experiment": "adversarial_pilot",
        "dataset": {
            "name": "cicids2017-flow",
            "n_benign": int(len(benign)),
            "attacks": {
                name: int(len(rows)) for name, rows in selected.items()
            },
            "seed": seed,
            "attack_factor": ATTACK_FACTOR,
            "fpr": DEFAULT_FPR,
        },
        "variants": {},
        "runtime_seconds": 0.0,
    }

    for policy_name, n_masks in variants:
        policy = POLICIES[policy_name]
        real, attack_threshold = _thresholds(
            model,
            benign,
            train_min,
            train_max,
            policy=policy,
            n_masks=n_masks,
            seed=seed,
            batch_size=batch_size,
        )
        variant_payload: dict[str, Any] = {
            "policy": policy_name,
            "n_masks": int(n_masks),
            "real_threshold": float(real),
            "attack_threshold": float(attack_threshold),
            "attacks": {},
        }
        cost = ScoringCost()
        for attack_name, rows in selected.items():
            runner = FlowAttack(
                model,
                train_min,
                train_max,
                decimal,
                x.shape[1],
                real,
                attack_threshold,
                seed=seed,
                n_masks=n_masks,
                mask_policy=policy,
            )
            sample_rows = rows[:per_attack]
            original_scores = score_samples(
                model,
                x[sample_rows],
                train_min,
                train_max,
                mode="repo_plus",
                n_masks=n_masks,
                batch_size=batch_size,
                seed=seed,
                phase=PHASE_ORIGINAL,
                sample_indices=indices[sample_rows],
                mask_policy=policy,
                cost=cost,
            )
            outcomes = []
            victims = []
            for position, sample_index, original_score in zip(
                sample_rows, indices[sample_rows], original_scores
            ):
                outcome = runner.find_adv(x[position], int(sample_index))
                victim = (
                    x[position]
                    if outcome.x_adv is None
                    else np.asarray(outcome.x_adv, dtype=np.float32)
                )
                victims.append(victim)
                outcomes.append(
                    {
                        "sample_index": int(sample_index),
                        "kind": outcome.kind,
                        "score": None if outcome.score is None else float(outcome.score),
                        "model_calls": int(outcome.model_calls),
                        "masks_evaluated": int(outcome.masks_evaluated),
                    }
                )
            victim_scores = score_samples(
                model,
                np.stack(victims),
                train_min,
                train_max,
                mode="repo_plus",
                n_masks=n_masks,
                batch_size=batch_size,
                seed=seed,
                phase=PHASE_VICTIM,
                sample_indices=indices[sample_rows],
                mask_policy=policy,
                cost=cost,
            )
            kinds = [item["kind"] for item in outcomes]
            normal_tpr = float(np.mean(original_scores >= attack_threshold))
            victim_tpr = float(np.mean(victim_scores >= attack_threshold))
            variant_payload["attacks"][attack_name] = {
                "n": int(len(sample_rows)),
                "normal_tpr": normal_tpr,
                "victim_tpr": victim_tpr,
                "fooled": int(sum(kind in ("evaded", "backup") for kind in kinds)),
                "kinds": {
                    kind: int(sum(item == kind for item in kinds))
                    for kind in ("evaded", "backup", "no_change", "not_fooled")
                },
                "outcomes": outcomes,
            }
            print(
                f"{policy_name}-{n_masks} {attack_name}: "
                f"victim_tpr={victim_tpr:.3f} fooled="
                f"{variant_payload['attacks'][attack_name]['fooled']}/{len(kinds)} "
                f"t={time.perf_counter() - started:.0f}s"
            )
        variant_payload["cost"] = cost.as_dict()
        payload["variants"][f"{policy_name}-{n_masks}"] = variant_payload

    payload["runtime_seconds"] = time.perf_counter() - started
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "flw_model"))
    parser.add_argument("--cache", default="")
    parser.add_argument("--max-benign", type=int, default=5_000)
    parser.add_argument("--max-per-attack", type=int, default=300)
    parser.add_argument("--per-attack", type=int, default=20)
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--variants", default="iid:100,iid:25,balanced:25,covering:25")
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "mask_efficiency" / "adversarial_pilot.json"
        ),
    )
    return parser


def _parse_variants(value: str) -> list[tuple[str, int]]:
    variants = []
    for item in value.split(","):
        policy, _, count = item.partition(":")
        variants.append((policy.strip(), int(count)))
    return variants


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_adversarial(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        per_attack=args.per_attack,
        attacks=[item.strip() for item in args.attacks.split(",") if item.strip()],
        variants=_parse_variants(args.variants),
        seed=args.seed,
        batch_size=args.batch_size,
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
