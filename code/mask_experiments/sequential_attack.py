"""Adversarial evaluation of the sequential RePO rule (Direction B).

The sequential rule (:func:`new_data_eval.decision.sequential_repo_decisions`)
reveals masks in blocks and stops as soon as the confidence interval of the
mean error excludes the threshold.  Its attack surface is the **earliest stop**:
an attacker who keeps the first block's upper confidence bound below the
threshold makes the rule declare the sample benign after ``block`` masks.

This module mirrors :mod:`mask_experiments.cascade_attack`:

* the attacker minimizes the statistic the rule compares with its threshold
  (``mean + z * standard_error`` of the block, see
  :func:`new_data_eval.attack.ci_objective`);
* the victim decision is recomputed by the real rule over 100 **fresh** masks,
  so the reported ``victim_tpr`` is the decision of the deployed policy, not of
  the attack's own criterion;
* every attack is evaluated against the same benign sample, the same threshold
  and the same attack rows as the cascade study.

Alongside the sequential numbers the runner stores the transfer of the same
victims to the two references of the paper: the 100-mask mean-based RePO
(``transfer_full100_tpr``) and the 100-mask RePO+ (``repo_plus_full100_tpr``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mask_experiments.data import BENIGN_LABEL, FLOW_ATTACKS, load_flow_meta
from mask_experiments.runner import REPO_ROOT, load_flow_model
from mask_experiments.variance import _macro_tpr, load_sample
from new_data_eval.decision import SequentialResult, sequential_repo_decisions
from new_data_eval.masking import POLICIES
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import (
    PHASE_ORIGINAL,
    PHASE_VICTIM,
    ScoringCost,
    reconstruction_errors,
    repo_plus_scores,
    repo_scores,
    threshold_at_fpr,
)

__all__ = [
    "DEFAULT_ATTACKS",
    "DEFAULT_BLOCKS",
    "DEFAULT_FPR",
    "DEFAULT_MAX_MASKS",
    "DEFAULT_Z",
    "run_sequential_attack",
    "sequential_attack_metrics",
    "validate_blocks",
]

DEFAULT_ATTACKS: tuple[str, ...] = FLOW_ATTACKS
DEFAULT_BLOCKS: tuple[int, ...] = (5, 10, 25)
DEFAULT_FPR = 0.10
DEFAULT_Z = 1.96
DEFAULT_MAX_MASKS = 100


def validate_blocks(
    blocks: Iterable[int], max_masks: int
) -> tuple[int, ...]:
    """Return ``blocks`` as a tuple of positive divisors of ``max_masks``.

    A block that does not divide the budget would make the rule stop at the
    last multiple while the attacker optimizes a block the rule never uses, so
    the mismatch is rejected instead of silently changing the protocol.
    """
    values = tuple(int(block) for block in blocks)
    if not values:
        raise ValueError("blocks must not be empty")
    budget = int(max_masks)
    if budget <= 0:
        raise ValueError(f"max_masks must be positive, got {max_masks!r}")
    for block in values:
        if block <= 0:
            raise ValueError(f"every block must be positive, got {block}")
        if block > budget:
            raise ValueError(
                f"block {block} exceeds max_masks ({budget})"
            )
        if budget % block != 0:
            raise ValueError(
                f"every block must divide max_masks ({budget}); got {block}"
            )
    return values


def restrict_result(
    result: SequentialResult, rows: np.ndarray
) -> SequentialResult:
    """Return the decisions of ``result`` restricted to ``rows``.

    The runner keeps the normal decisions of the whole sample while the
    victims only exist for the attacked rows; slicing the normal result first
    makes both arrays share the same local index.
    """
    selection = np.asarray(rows, dtype=np.int64)
    return SequentialResult(
        decision=np.asarray(result.decision, dtype=bool)[selection],
        masks_used=np.asarray(result.masks_used, dtype=np.int64)[selection],
    )


def sequential_attack_metrics(
    *,
    normal: SequentialResult,
    victim: SequentialResult,
    rows: np.ndarray,
    block: int,
    transfer_tpr: float,
    repo_plus_tpr: float,
) -> dict[str, Any]:
    """Summarize the paired normal/victim decisions of the sequential rule.

    ``rows`` selects the attacked samples.  ``early_stop_rate`` is the fraction
    of victims the rule discarded as benign at the **first** block: that is the
    quantity the attack targets, while ``victim_tpr`` is the decision of the
    full rule (which may keep revealing masks).
    """
    selection = np.asarray(rows, dtype=np.int64)
    normal_attack = np.asarray(normal.decision, dtype=bool)[selection]
    victim_attack = np.asarray(victim.decision, dtype=bool)[selection]
    victim_masks = np.asarray(victim.masks_used, dtype=np.int64)[selection]
    normal_masks = np.asarray(normal.masks_used, dtype=np.int64)[selection]

    count = int(selection.size)
    normal_tpr = float(normal_attack.mean()) if count else 0.0
    victim_tpr = float(victim_attack.mean()) if count else 0.0
    early_stop = (~victim_attack) & (victim_masks == int(block))

    return {
        "n": count,
        "normal_tpr": normal_tpr,
        "victim_tpr": victim_tpr,
        "conditional_drop": (
            None if normal_tpr == 0.0 else float(1.0 - victim_tpr / normal_tpr)
        ),
        "fooled": int(np.sum(normal_attack & ~victim_attack)),
        "already_benign": int(np.sum(~normal_attack)),
        "early_stop_rate": float(early_stop.mean()) if count else 0.0,
        "victim_masks_used": float(victim_masks.mean()) if count else 0.0,
        "normal_masks_used": float(normal_masks.mean()) if count else 0.0,
        "transfer_full100_tpr": float(transfer_tpr),
        "repo_plus_full100_tpr": float(repo_plus_tpr),
    }


def _select_rows(
    labels: np.ndarray,
    attacks: Iterable[str],
    *,
    per_attack: int,
    generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return the sorted global positions of the attacked samples."""
    selected: dict[str, np.ndarray] = {}
    for attack in attacks:
        positions = np.flatnonzero(labels == attack)
        take = min(int(per_attack), len(positions))
        if take:
            selected[attack] = np.sort(
                generator.choice(positions, size=take, replace=False)
            )
    return selected


def run_sequential_attack(
    *,
    data_dir: Path,
    model_dir: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    per_attack: int = 20,
    attacks: Iterable[str] = DEFAULT_ATTACKS,
    blocks: Iterable[int] = DEFAULT_BLOCKS,
    fpr: float = DEFAULT_FPR,
    z: float = DEFAULT_Z,
    max_masks: int = DEFAULT_MAX_MASKS,
    loss: str = "ci",
    seed: int = 2019,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Run the adaptive attack against the sequential rule on every attack."""
    # TensorFlow is imported only here: the module stays usable (and testable)
    # without the deep learning stack.
    from new_data_eval.attack import FlowAttack

    started = time.perf_counter()
    block_values = validate_blocks(blocks, max_masks)
    attack_names = [str(name) for name in attacks]

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
        n_masks=max_masks,
        batch_size=batch_size,
        seed=seed,
        phase=PHASE_ORIGINAL,
        sample_indices=indices,
        mask_policy=policy,
        cost=cost,
    )
    mean_scores = repo_scores(errors)
    plus_scores = repo_plus_scores(errors)
    decimal = [
        index
        for index in range(x.shape[1])
        if np.sum(np.abs(x[:, index] - np.round(x[:, index]))) == 0
    ]

    selected = _select_rows(
        labels, attack_names, per_attack=per_attack, generator=generator
    )

    # The attack protocol of the study: the "real" threshold is the one the
    # benign sample supports at `fpr`, and the attack is scored against the
    # very same value -- no threshold is relaxed for the victim.
    threshold = threshold_at_fpr(mean_scores[benign], fpr)
    plus_threshold = threshold_at_fpr(plus_scores[benign], fpr)

    payload: dict[str, Any] = {
        "schema_version": "mask_exp_v1",
        "experiment": "sequential_attack",
        "dataset": {
            "name": "cicids2017-flow",
            "n_benign": int(benign.sum()),
            "attacks": {name: int(len(rows)) for name, rows in selected.items()},
            "seed": seed,
            "fpr": float(fpr),
        },
        "configs": {},
        "attacks": {},
        "cost": {},
        "runtime_seconds": 0.0,
    }

    for block in block_values:
        normal = sequential_repo_decisions(
            errors,
            threshold=threshold,
            block=block,
            min_masks=block,
            max_masks=max_masks,
        )
        payload["configs"][f"seq{block}"] = {
            "block": int(block),
            "min_masks": int(block),
            "z": float(z),
            "max_masks": int(max_masks),
            "loss": loss,
            "threshold": float(threshold),
            "repo_plus_threshold": float(plus_threshold),
            "normal_fpr": float(normal.decision[benign].mean()),
            "normal_tpr_macro": _macro_tpr(
                normal.decision.astype(float), labels, 0.5
            )["macro"],
            "normal_masks_used_benign": float(normal.masks_used[benign].mean()),
            "normal_masks_used_attack": float(normal.masks_used[~benign].mean()),
        }

        for attack_name, rows in selected.items():
            # `FlowAttack.score` returns the RePO+ sum over `block` masks,
            # which is `block * mean`; scaling the rule's threshold by the
            # block size keeps both criteria in the same units.
            runner = FlowAttack(
                model,
                train_min,
                train_max,
                decimal,
                x.shape[1],
                block * threshold,
                block * threshold,
                seed=seed,
                n_masks=block,
                mask_policy=policy,
                loss=loss,
                ci_z=z,
            )
            outcomes = []
            victims = []
            for position in rows:
                outcome = runner.find_adv(x[position], int(indices[position]))
                victims.append(
                    x[position]
                    if outcome.x_adv is None
                    else np.asarray(outcome.x_adv, dtype=np.float32)
                )
                outcomes.append(outcome)

            errors_victim = reconstruction_errors(
                model,
                np.stack(victims),
                train_min,
                train_max,
                n_masks=max_masks,
                batch_size=batch_size,
                seed=seed,
                phase=PHASE_VICTIM,
                sample_indices=indices[rows],
                mask_policy=policy,
                cost=cost,
            )
            victim = sequential_repo_decisions(
                errors_victim,
                threshold=threshold,
                block=block,
                min_masks=block,
                max_masks=max_masks,
            )
            result = sequential_attack_metrics(
                normal=restrict_result(normal, rows),
                victim=victim,
                rows=np.arange(len(rows)),
                block=block,
                transfer_tpr=float(np.mean(repo_scores(errors_victim) >= threshold)),
                repo_plus_tpr=float(
                    np.mean(repo_plus_scores(errors_victim) >= plus_threshold)
                ),
            )
            result["kinds"] = {
                kind: int(sum(outcome.kind == kind for outcome in outcomes))
                for kind in ("evaded", "backup", "no_change", "not_fooled")
            }
            result["masks_evaluated"] = (
                int(np.mean([outcome.masks_evaluated for outcome in outcomes]))
                if outcomes
                else 0
            )
            payload["attacks"].setdefault(attack_name, {})[
                f"seq{block}"
            ] = result
            print(
                f"sequential-attack block={block} {attack_name}: "
                f"victim_tpr={result['victim_tpr']:.2f} "
                f"early_stop={result['early_stop_rate']:.2f} "
                f"masks={result['victim_masks_used']:.1f} "
                f"fooled={result['fooled']}/{result['n']}"
            )

    payload["cost"] = cost.as_dict()
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
    parser.add_argument(
        "--blocks", default=",".join(str(block) for block in DEFAULT_BLOCKS)
    )
    parser.add_argument("--z", type=float, default=DEFAULT_Z)
    parser.add_argument("--max-masks", type=int, default=DEFAULT_MAX_MASKS)
    parser.add_argument("--loss", default="ci", choices=("sum", "ci"))
    parser.add_argument("--fpr", type=float, default=DEFAULT_FPR)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "mask_efficiency" / "sequential_attack.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_sequential_attack(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        per_attack=args.per_attack,
        attacks=[item.strip() for item in args.attacks.split(",") if item.strip()],
        blocks=[int(value) for value in args.blocks.split(",") if value.strip()],
        fpr=args.fpr,
        z=args.z,
        max_masks=args.max_masks,
        loss=args.loss,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
