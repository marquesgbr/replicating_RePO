"""Runner of the mask efficiency experiments.

The first experiment is the frozen baseline: score a uniform random sample of
the CICIDS2017 flow test set with the reference protocol (100 IID masks, 75 %
masked, seed 2019) and archive the scores, rates and cost.  Later experiments
(variance curves, adversarial evaluations, cascades, distillation) reuse the
same payload builder and JSON schema.

TensorFlow is imported only by :func:`load_flow_model`, so the aggregation and
the tests run without the deep learning stack.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mask_experiments.data import (
    BENIGN_LABEL,
    FLOW_ATTACKS,
    FLOW_TEST_DAYS,
    load_flow_meta,
    sample_flow_test,
)
from new_data_eval.masking import IIDMaskPolicy, MaskPolicy
from new_data_eval.results import (
    config_hash,
    environment_manifest,
    git_commit,
    write_json_atomic,
)
from new_data_eval.scoring import ScoringCost, score_samples, threshold_at_fpr

__all__ = [
    "SCHEMA_VERSION",
    "build_baseline_payload",
    "class_rates",
    "load_flow_model",
    "main",
    "score_matrix",
]

#: Version of the experiment payload schema.
SCHEMA_VERSION = "mask_exp_v1"

#: Target false positive rates reported by the baseline.
FPRS: tuple[float, ...] = (0.01, 0.10)

#: Repository root (``code/mask_experiments`` -> repository).
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_flow_model(model_dir: str | Path) -> Any:
    """Load the flow SavedModel with the same adapter used by the notebooks."""
    code_dir = REPO_ROOT / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from load_model_compat import load_model

    return load_model(str(Path(model_dir).expanduser().resolve()))


def _directory_sha256(path: Path) -> str:
    """Hash the file names and contents of a SavedModel directory."""
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def class_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return the per-class TPR (BENIGN reports its FPR) at ``threshold``."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    rates: dict[str, dict[str, Any]] = {}
    for name in class_names:
        mask = labels == name
        count = int(np.count_nonzero(mask))
        rate = float(np.mean(scores[mask] >= threshold)) if count else 0.0
        item: dict[str, Any] = {"n": count, "rate": rate}
        if name == BENIGN_LABEL:
            item["fpr"] = rate
        rates[name] = item
    return rates


def _average_attack_tpr(
    rates: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Mean TPR over the reference flow attacks (Web Attack excluded)."""
    attacks = [
        (name, rates[name])
        for name in FLOW_ATTACKS
        if name in rates and int(rates[name].get("n", 0)) > 0
    ]
    if not attacks:
        return {"macro": 0.0, "weighted": 0.0}
    tprs = np.array([float(item["rate"]) for _, item in attacks])
    counts = np.array([int(item["n"]) for _, item in attacks])
    return {
        "macro": float(tprs.mean()),
        "weighted": float(np.average(tprs, weights=counts)),
    }


def score_matrix(
    model: Any,
    x_raw: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    modes: Sequence[str],
    n_masks: int,
    batch_size: int,
    seed: int,
    score_repeats: int,
    sample_indices: np.ndarray,
    policy: MaskPolicy | None = None,
    cost: ScoringCost | None = None,
) -> dict[str, np.ndarray]:
    """Score every sample in every requested mode (RePO and/or RePO+)."""
    scores: dict[str, np.ndarray] = {}
    for mode in modes:
        scores[mode] = score_samples(
            model,
            x_raw,
            train_min,
            train_max,
            mode=mode,
            n_masks=n_masks,
            batch_size=batch_size,
            seed=seed,
            sample_indices=sample_indices,
            score_repeats=score_repeats,
            mask_policy=policy,
            cost=cost,
        )
    return scores


def build_baseline_payload(
    *,
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    x_raw: np.ndarray,
    sample_indices: np.ndarray,
    dataset: Mapping[str, Any],
    model_dir: Path,
    n_masks: int,
    policy_name: str,
    seed: int,
    score_repeats: int,
    batch_size: int,
    cost: ScoringCost,
    runtime_seconds: float,
    fprs: Sequence[float] = FPRS,
) -> dict[str, Any]:
    """Assemble the JSON document of the frozen baseline."""
    labels = np.asarray(labels).ravel()
    class_names = [BENIGN_LABEL] + sorted(
        {str(label) for label in labels if str(label) != BENIGN_LABEL}
    )
    counts = {name: int(np.count_nonzero(labels == name)) for name in class_names}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "baseline",
        "dataset": {
            **dataset,
            "n_samples": int(len(labels)),
            "counts": counts,
            "n_features": int(x_raw.shape[1]),
        },
        "model": {
            "path": str(model_dir),
            "sha256": _directory_sha256(model_dir),
        },
        "scoring": {
            "n_masks": int(n_masks),
            "policy": str(policy_name),
            "seed": int(seed),
            "score_repeats": int(score_repeats),
            "batch_size": int(batch_size),
            "mask_probability": 0.75,
        },
        "cost": cost.as_dict(),
        "runtime_seconds": float(runtime_seconds),
        "environment": environment_manifest(),
        "git_commit": git_commit(),
    }

    mode_payload: dict[str, Any] = {}
    for mode, mode_scores in scores.items():
        thresholds = {
            f"fpr_{fpr}": threshold_at_fpr(mode_scores[labels == BENIGN_LABEL], fpr)
            for fpr in fprs
        }
        rates = {
            name: class_rates(mode_scores, labels, threshold, class_names)
            for name, threshold in thresholds.items()
        }
        mode_payload[mode] = {
            "thresholds": thresholds,
            "fpr": {
                name: rates[name][BENIGN_LABEL]["fpr"] for name in thresholds
            },
            "class_rates": rates,
            "average_tpr": {
                name: _average_attack_tpr(rates[name]) for name in thresholds
            },
        }
    payload["modes"] = mode_payload

    run_parameters = {
        "dataset": {
            key: value for key, value in payload["dataset"].items() if key != "n_samples"
        },
        "model": payload["model"],
        "scoring": payload["scoring"],
        "git_commit": payload["git_commit"],
    }
    payload["config_hash"] = config_hash(run_parameters)
    return payload


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--modes must list at least one mode")
    unknown = [mode for mode in modes if mode not in ("repo", "repo_plus")]
    if unknown:
        raise ValueError(f"unknown mode(s): {unknown}")
    return list(dict.fromkeys(modes))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen baseline of the RePO mask efficiency study."
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument(
        "--model-dir", default=str(REPO_ROOT / "models" / "flw_model")
    )
    parser.add_argument("--max-benign", type=int, default=2_000)
    parser.add_argument("--max-per-attack", type=int, default=500)
    parser.add_argument("--n-masks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--score-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--modes", default="repo,repo_plus")
    parser.add_argument("--days", default=",".join(FLOW_TEST_DAYS))
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "mask_efficiency" / "baseline_flow.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    modes = _parse_modes(args.modes)
    if "repo_plus" in modes and args.n_masks % 5 != 0:
        raise SystemExit("--n-masks must be divisible by 5 for repo_plus")
    days = tuple(item.strip() for item in args.days.split(",") if item.strip())

    started = time.perf_counter()
    data_dir = Path(args.data_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    x_raw, labels, sample_indices = sample_flow_test(
        data_dir,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        seed=args.seed,
        days=days,
    )
    train_min, train_max = load_flow_meta(data_dir)
    model = load_flow_model(model_dir)

    cost = ScoringCost()
    policy = IIDMaskPolicy()
    scores = score_matrix(
        model,
        x_raw,
        train_min,
        train_max,
        modes=modes,
        n_masks=args.n_masks,
        batch_size=args.batch_size,
        seed=args.seed,
        score_repeats=args.score_repeats,
        sample_indices=sample_indices,
        policy=policy,
        cost=cost,
    )
    payload = build_baseline_payload(
        scores=scores,
        labels=labels,
        x_raw=x_raw,
        sample_indices=sample_indices,
        dataset={
            "name": "cicids2017-flow",
            "days": list(days),
            "sampling": {
                "seed": args.seed,
                "max_benign": args.max_benign,
                "max_per_attack": args.max_per_attack,
            },
        },
        model_dir=model_dir,
        n_masks=args.n_masks,
        policy_name=policy.name,
        seed=args.seed,
        score_repeats=args.score_repeats,
        batch_size=args.batch_size,
        cost=cost,
        runtime_seconds=time.perf_counter() - started,
    )
    write_json_atomic(Path(args.out), payload)
    for mode, item in payload["modes"].items():
        print(
            f"{mode}: n={payload['dataset']['n_samples']} "
            f"fpr_0.01={item['fpr']['fpr_0.01']:.4f} "
            f"tpr_weighted={item['average_tpr']['fpr_0.01']['weighted']:.4f} "
            f"cost={payload['cost']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
