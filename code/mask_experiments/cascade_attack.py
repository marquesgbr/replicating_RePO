"""Adversarial evaluation of the cascade (Direction B).

The attacker targets the **cheap stage**: it optimizes the RePO+ score of the
first ``n_cheap`` masks until it falls below ``t_low``.  If it succeeds, the
sample is declared benign without the full ensemble -- the cascade is only as
strong as its cheap stage.

The helper also reports the paired normal/victim decisions of the full cascade
and the bypass rate, which is the quantity a reviewer will ask for.
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
from new_data_eval.decision import (
    cascade_decisions,
    cascade_threshold,
    cheap_repo_plus,
)
from new_data_eval.masking import POLICIES
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import (
    PHASE_ORIGINAL,
    PHASE_VICTIM,
    ScoringCost,
    reconstruction_errors,
    score_samples,
    threshold_at_fpr,
)

__all__ = ["DEFAULT_ATTACKS", "run_cascade_attack"]

DEFAULT_ATTACKS: tuple[str, ...] = (
    "FTP-Patator",
    "DoS slowloris",
    "DoS Hulk",
    "DDoS",
)

DEFAULT_CHEAP: tuple[int, ...] = (5, 25)
DEFAULT_FN_RATE = 0.01


def _decide(
    cheap: np.ndarray, full: np.ndarray, t_low: float, threshold: float
) -> np.ndarray:
    return (cheap >= t_low) & (full >= threshold)


def run_cascade_attack(
    *,
    data_dir: Path,
    model_dir: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    per_attack: int = 20,
    attacks: Iterable[str] = DEFAULT_ATTACKS,
    cheap_options: Iterable[int] = DEFAULT_CHEAP,
    fn_rate: float = DEFAULT_FN_RATE,
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
    policy = POLICIES["iid"]
    benign = labels == BENIGN_LABEL
    generator = np.random.default_rng(seed + 4)

    cost = ScoringCost()
    errors = reconstruction_errors(
        model,
        x,
        train_min,
        train_max,
        n_masks=100,
        batch_size=batch_size,
        seed=seed,
        phase=PHASE_ORIGINAL,
        sample_indices=indices,
        mask_policy=policy,
        cost=cost,
    )
    decimal = [
        index
        for index in range(x.shape[1])
        if np.sum(np.abs(x[:, index] - np.round(x[:, index]))) == 0
    ]

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
        "experiment": "cascade_attack",
        "dataset": {
            "name": "cicids2017-flow",
            "n_benign": int(benign.sum()),
            "attacks": {name: int(len(rows)) for name, rows in selected.items()},
            "seed": seed,
            "false_negative_rate_target": fn_rate,
        },
        "configs": {},
        "attacks": {},
        "runtime_seconds": 0.0,
    }

    for n_cheap in cheap_options:
        cheap = cheap_repo_plus(errors, n_cheap)
        t_low = cascade_threshold(
            cheap[~benign], false_negative_rate=fn_rate
        )
        probe = cascade_decisions(
            errors, n_cheap=n_cheap, t_low=t_low, threshold=0.0
        )
        threshold = threshold_at_fpr(probe.score[benign], 0.01)
        full_original = cheap_repo_plus(errors, errors.shape[1])
        normal = _decide(cheap, full_original, t_low, threshold)
        config = {
            "n_cheap": int(n_cheap),
            "t_low": float(t_low),
            "threshold": float(threshold),
            "normal_tpr": _macro_tpr(normal.astype(float), labels, 0.5),
            "benign_skip_rate": float((~probe.routed[benign]).mean()),
        }

        for attack_name, rows in selected.items():
            runner = FlowAttack(
                model,
                train_min,
                train_max,
                decimal,
                x.shape[1],
                t_low,
                t_low,
                seed=seed,
                n_masks=n_cheap,
                mask_policy=policy,
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

            cheap_victim = score_samples(
                model,
                np.stack(victims),
                train_min,
                train_max,
                mode="repo_plus",
                n_masks=n_cheap,
                batch_size=batch_size,
                seed=seed,
                phase=PHASE_VICTIM,
                sample_indices=indices[rows],
                mask_policy=policy,
                cost=cost,
            )
            full_victim = score_samples(
                model,
                np.stack(victims),
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
            victim = _decide(cheap_victim, full_victim, t_low, threshold)
            normal_attack = normal[rows]
            bypass = cheap_victim < t_low
            result = {
                "n": int(len(rows)),
                "normal_tpr": float(normal_attack.mean()),
                "victim_tpr": float(victim.mean()),
                "conditional_drop": (
                    None
                    if normal_attack.mean() == 0.0
                    else float(1.0 - victim.mean() / normal_attack.mean())
                ),
                "bypass_rate": float(bypass.mean()),
                "fooled": int(np.sum(normal_attack & ~victim)),
                "already_benign": int(np.sum(~normal_attack)),
                "kinds": {
                    kind: int(sum(outcome.kind == kind for outcome in outcomes))
                    for kind in ("evaded", "backup", "no_change", "not_fooled")
                },
                "masks_evaluated": int(
                    np.mean([outcome.masks_evaluated for outcome in outcomes])
                ),
            }
            payload["attacks"].setdefault(attack_name, {})[
                f"cheap{int(n_cheap)}"
            ] = result
            print(
                f"cascade-attack cheap={n_cheap} {attack_name}: "
                f"victim_tpr={result['victim_tpr']:.2f} "
                f"bypass={result['bypass_rate']:.2f} "
                f"fooled={result['fooled']}/{result['n']}"
            )

        payload["configs"][f"cheap{int(n_cheap)}"] = config

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
    parser.add_argument("--cheap", default="5,25")
    parser.add_argument("--fn-rate", type=float, default=DEFAULT_FN_RATE)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "mask_efficiency" / "cascade_attack.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_cascade_attack(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        per_attack=args.per_attack,
        attacks=[item.strip() for item in args.attacks.split(",") if item.strip()],
        cheap_options=[int(value) for value in args.cheap.split(",")],
        fn_rate=args.fn_rate,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
