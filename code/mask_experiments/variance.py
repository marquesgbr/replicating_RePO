"""Variance study of the RePO mask ensemble (Task 1, Direction A).

For every mask policy and every ``n_masks`` the study measures, on a uniform
random sample of the CICIDS2017 flow test set:

* the dispersion of the RePO/RePO+ scores across independent mask sets;
* the rank agreement with the reference 100-mask IID score (Spearman);
* the TPR at the matched FPR (threshold recalibrated on the benign scores);
* the cost in masked rows.

The output is one JSON document plus a flat CSV, ready for the aggregator.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from mask_experiments.data import (
    BENIGN_LABEL,
    FLOW_ATTACKS,
    FLOW_TEST_DAYS,
    load_flow_meta,
    sample_flow_test,
)
from mask_experiments.runner import REPO_ROOT, load_flow_model
from new_data_eval.masking import POLICIES, MaskPolicy
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import (
    ScoringCost,
    reconstruction_error_repeats,
    score_from_errors,
    threshold_at_fpr,
)

__all__ = [
    "DEFAULT_N_MASKS",
    "DEFAULT_POLICIES",
    "DEFAULT_REPEATS",
    "load_sample",
    "run_variance",
]

DEFAULT_N_MASKS: tuple[int, ...] = (5, 10, 20, 25, 50, 100)
DEFAULT_POLICIES: tuple[str, ...] = ("iid", "balanced", "covering")
DEFAULT_REPEATS = 3
DEFAULT_FPRS: tuple[float, ...] = (0.01, 0.10)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without SciPy."""
    ra, rb = _rank(np.asarray(a)), _rank(np.asarray(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denominator == 0.0:
        return 0.0
    return float((ra * rb).sum() / denominator)


def _macro_tpr(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    tprs = []
    counts = []
    for name in FLOW_ATTACKS:
        mask = labels == name
        count = int(mask.sum())
        if count == 0:
            continue
        tprs.append(float(np.mean(scores[mask] >= threshold)))
        counts.append(count)
    if not tprs:
        return {"macro": 0.0, "weighted": 0.0}
    tprs_array = np.array(tprs)
    counts_array = np.array(counts)
    return {
        "macro": float(tprs_array.mean()),
        "weighted": float(np.average(tprs_array, weights=counts_array)),
    }


def _class_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    rates = {}
    for name in class_names:
        mask = labels == name
        count = int(mask.sum())
        rate = float(np.mean(scores[mask] >= threshold)) if count else 0.0
        rates[name] = {"n": count, "rate": rate}
    return rates


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"], data["labels"].astype(str), data["indices"]


def load_sample(
    data_dir: Path,
    *,
    cache: Path | None,
    max_benign: int,
    max_per_attack: int,
    seed: int,
    days: Iterable[str] = FLOW_TEST_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the evaluation sample (from the cache when it exists)."""
    if cache is not None and cache.is_file():
        x, labels, indices = _load_npz(cache)
    else:
        x, labels, indices = sample_flow_test(
            data_dir,
            max_benign=max_benign,
            max_per_attack=max_per_attack,
            seed=seed,
            days=tuple(days),
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, x=x, labels=labels, indices=indices)

    # Deterministic subsample of the cached (already random) sample.
    generator = np.random.default_rng(seed + 1)
    keep: list[np.ndarray] = []
    for name in [BENIGN_LABEL] + sorted(set(labels) - {BENIGN_LABEL}):
        positions = np.flatnonzero(labels == name)
        cap = max_benign if name == BENIGN_LABEL else max_per_attack
        if len(positions) > cap:
            positions = np.sort(generator.choice(positions, size=cap, replace=False))
        keep.append(positions)
    selected = np.sort(np.concatenate(keep))
    return x[selected], labels[selected], indices[selected]


def _score_tensor(
    model: Any,
    x_raw: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    policy: MaskPolicy,
    n_masks: int,
    seed: int,
    repeats: int,
    sample_indices: np.ndarray,
    batch_size: int,
    cost: ScoringCost,
) -> dict[str, np.ndarray]:
    errors = reconstruction_error_repeats(
        model,
        x_raw,
        train_min,
        train_max,
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        sample_indices=sample_indices,
        score_repeats=repeats,
        mask_policy=policy,
        cost=cost,
    )
    return {
        mode: np.stack(
            [score_from_errors(errors[repeat], mode) for repeat in range(repeats)]
        )
        for mode in ("repo", "repo_plus")
    }


def _evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fprs: tuple[float, ...],
    reference: np.ndarray | None,
) -> dict[str, Any]:
    benign = labels == BENIGN_LABEL
    result: dict[str, Any] = {
        "score_mean": float(scores.mean()),
        "score_std_across_repeats": float(scores.std(axis=0, ddof=1).mean()),
        "score_std_p95": float(np.percentile(scores.std(axis=0, ddof=1), 95)),
        "spearman_vs_reference": (
            None if reference is None else spearman(scores.mean(axis=0), reference)
        ),
    }
    class_names = [BENIGN_LABEL] + [
        name for name in FLOW_ATTACKS if (labels == name).any()
    ]
    thresholds_payload = {}
    for fpr in fprs:
        name = f"fpr_{fpr}"
        thresholds = np.array(
            [threshold_at_fpr(scores[repeat][benign], fpr) for repeat in range(len(scores))]
        )
        # The deployment uses one mask set, so the reported TPR is the mean of
        # the per-repeat TPRs -- not the TPR of the repeat-averaged score, which
        # would be an ensemble of `repeats * n_masks` masks.
        macros = [
            _macro_tpr(scores[repeat], labels, float(thresholds[repeat]))
            for repeat in range(len(scores))
        ]
        class_runs = [
            _class_rates(
                scores[repeat], labels, float(thresholds[repeat]), class_names
            )
            for repeat in range(len(scores))
        ]
        class_rates = {
            class_name: {
                "n": class_runs[0][class_name]["n"],
                "rate": float(
                    np.mean([run[class_name]["rate"] for run in class_runs])
                ),
            }
            for class_name in class_names
        }
        thresholds_payload[name] = {
            "threshold_mean": float(thresholds.mean()),
            "threshold_std": float(thresholds.std()),
            "tpr": {
                "macro": float(np.mean([item["macro"] for item in macros])),
                "macro_std": float(np.std([item["macro"] for item in macros])),
                "weighted": float(np.mean([item["weighted"] for item in macros])),
                "weighted_std": float(
                    np.std([item["weighted"] for item in macros])
                ),
            },
            "class_rates": class_rates,
        }
    result["thresholds"] = thresholds_payload
    return result


def run_variance(
    *,
    data_dir: Path,
    model_dir: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    n_masks: Iterable[int] = DEFAULT_N_MASKS,
    policies: Iterable[str] = DEFAULT_POLICIES,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 2019,
    batch_size: int = 100,
    fprs: tuple[float, ...] = DEFAULT_FPRS,
    days: Iterable[str] = FLOW_TEST_DAYS,
) -> dict[str, Any]:
    started = time.perf_counter()
    x, labels, sample_indices = load_sample(
        data_dir,
        cache=cache,
        max_benign=max_benign,
        max_per_attack=max_per_attack,
        seed=seed,
        days=days,
    )
    train_min, train_max = load_flow_meta(data_dir)
    model = load_flow_model(model_dir)

    payload: dict[str, Any] = {
        "schema_version": "mask_exp_v1",
        "experiment": "variance",
        "dataset": {
            "name": "cicids2017-flow",
            "days": list(days),
            "n_samples": int(len(labels)),
            "counts": {
                name: int((labels == name).sum())
                for name in [BENIGN_LABEL] + sorted(set(labels) - {BENIGN_LABEL})
            },
            "sampling": {
                "seed": seed,
                "max_benign": max_benign,
                "max_per_attack": max_per_attack,
            },
        },
        "scoring": {
            "n_masks": list(n_masks),
            "policies": list(policies),
            "score_repeats": repeats,
            "seed": seed,
            "batch_size": batch_size,
        },
        "results": {},
    }

    reference_scores: np.ndarray | None = None
    if "iid" in list(policies):
        reference_cost = ScoringCost()
        reference = _score_tensor(
            model,
            x,
            train_min,
            train_max,
            policy=POLICIES["iid"],
            n_masks=max(n_masks),
            seed=seed,
            repeats=repeats,
            sample_indices=sample_indices,
            batch_size=batch_size,
            cost=reference_cost,
        )
        reference_scores = reference["repo_plus"].mean(axis=0)

    for policy_name in policies:
        policy = POLICIES[policy_name]
        policy_payload: dict[str, Any] = {}
        for count in n_masks:
            cost = ScoringCost()
            scores = _score_tensor(
                model,
                x,
                train_min,
                train_max,
                policy=policy,
                n_masks=count,
                seed=seed,
                repeats=repeats,
                sample_indices=sample_indices,
                batch_size=batch_size,
                cost=cost,
            )
            policy_payload[str(count)] = {
                mode: _evaluate_scores(
                    mode_scores,
                    labels,
                    fprs=fprs,
                    reference=reference_scores,
                )
                for mode, mode_scores in scores.items()
            }
            policy_payload[str(count)]["cost"] = cost.as_dict()
            policy_payload[str(count)]["cost"]["samples"] = int(len(labels))
            print(
                f"policy={policy_name} n={count} "
                f"rows={cost.masked_rows} t={time.perf_counter() - started:.0f}s"
            )
        payload["results"][policy_name] = policy_payload

    payload["runtime_seconds"] = time.perf_counter() - started
    return payload


def write_csv(payload: Mapping[str, Any], path: Path) -> None:
    """Flatten the variance payload into one row per (policy, n, mode, fpr)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "policy",
                "n_masks",
                "mode",
                "fpr",
                "threshold",
                "tpr_macro",
                "tpr_weighted",
                "score_std",
                "spearman",
                "masked_rows",
                "samples",
            ]
        )
        for policy, by_n in payload["results"].items():
            for n_masks, item in by_n.items():
                for mode in ("repo", "repo_plus"):
                    for fpr, threshold in item[mode]["thresholds"].items():
                        writer.writerow(
                            [
                                policy,
                                n_masks,
                                mode,
                                fpr,
                                threshold["threshold_mean"],
                                threshold["tpr"]["macro"],
                                threshold["tpr"]["weighted"],
                                item[mode]["score_std_across_repeats"],
                                item[mode]["spearman_vs_reference"],
                                item["cost"]["masked_rows"],
                                item["cost"]["samples"],
                            ]
                        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "flw_model"))
    parser.add_argument("--cache", default="/tmp/opencode/mask_exp/sample_flow.npz")
    parser.add_argument("--max-benign", type=int, default=5_000)
    parser.add_argument("--max-per-attack", type=int, default=300)
    parser.add_argument("--n-masks", default="5,10,20,25,50,100")
    parser.add_argument("--policies", default="iid,balanced,covering")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "mask_efficiency" / "variance_flow.json"),
    )
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "results" / "mask_efficiency" / "variance_flow.csv"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_variance(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        n_masks=[int(value) for value in args.n_masks.split(",")],
        policies=[value.strip() for value in args.policies.split(",")],
        repeats=args.repeats,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    write_json_atomic(Path(args.out), payload)
    write_csv(payload, Path(args.csv))
    print(f"wrote {args.out} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
