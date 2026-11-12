"""Cascade and sequential scoring pilot (Task 3, Direction B).

The full 100-mask error matrix is computed once.  The cheap rules are then
applied to the matrix and the *number of masks they would use* is reported per
sample, so the cost of each rule is measured without re-running the model.

A cascade is useful only if the cheap stage is not the weakest link: an
attacker that brings the cheap score below ``t_low`` bypasses the ensemble.
The pilot therefore reports the false negatives of the cheap stage and the
benign skip rate for several budgets, and the adversarial pilot includes an
``iid-5`` variant as a proxy for attacking a 5-mask cheap stage.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mask_experiments.data import BENIGN_LABEL, FLOW_ATTACKS, load_flow_meta
from mask_experiments.runner import REPO_ROOT, load_flow_model
from mask_experiments.variance import load_sample
from new_data_eval.decision import (
    cascade_decisions,
    cascade_threshold,
    cheap_repo_plus,
    sequential_repo_decisions,
)
from new_data_eval.masking import POLICIES
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import (
    ScoringCost,
    reconstruction_error_repeats,
    score_from_errors,
    threshold_at_fpr,
)

__all__ = ["run_cascade"]

DEFAULT_CHEAP: tuple[int, ...] = (5, 10, 25)
DEFAULT_FN_RATES: tuple[float, ...] = (0.0, 0.001, 0.01)
BENIGN_FRACTION = 0.99


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


def _mix_masks(
    masks_used: np.ndarray, labels: np.ndarray, *, benign_fraction: float = BENIGN_FRACTION
) -> float:
    benign = labels == BENIGN_LABEL
    attack = ~benign
    benign_cost = float(masks_used[benign].mean()) if benign.any() else 0.0
    attack_cost = float(masks_used[attack].mean()) if attack.any() else 0.0
    return benign_fraction * benign_cost + (1.0 - benign_fraction) * attack_cost


def run_cascade(
    *,
    data_dir: Path,
    model_dir: Path,
    cache: Path | None = None,
    max_benign: int = 5_000,
    max_per_attack: int = 300,
    n_masks: int = 100,
    repeats: int = 3,
    seed: int = 2019,
    batch_size: int = 100,
    cheap_options: Iterable[int] = DEFAULT_CHEAP,
    fn_rates: Iterable[float] = DEFAULT_FN_RATES,
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

    cost = ScoringCost()
    errors = reconstruction_error_repeats(
        model,
        x,
        train_min,
        train_max,
        n_masks=n_masks,
        batch_size=batch_size,
        seed=seed,
        sample_indices=indices,
        score_repeats=repeats,
        mask_policy=policy,
        cost=cost,
    )
    benign = labels == BENIGN_LABEL

    full_repo = np.stack(
        [score_from_errors(errors[repeat], "repo") for repeat in range(repeats)]
    )
    full_plus = np.stack(
        [score_from_errors(errors[repeat], "repo_plus") for repeat in range(repeats)]
    )

    payload: dict[str, Any] = {
        "schema_version": "mask_exp_v1",
        "experiment": "cascade",
        "dataset": {
            "name": "cicids2017-flow",
            "n_samples": int(len(labels)),
            "n_benign": int(benign.sum()),
            "n_attack": int((~benign).sum()),
            "seed": seed,
        },
        "scoring": {
            "n_masks": n_masks,
            "policy": policy.name,
            "score_repeats": repeats,
            "batch_size": batch_size,
        },
        "full_repo_plus": {},
        "cascade": {},
        "sequential_repo": {},
        "cost": cost.as_dict(),
        "runtime_seconds": 0.0,
    }

    # Reference: full RePO+ at the matched FPR.
    ref_thresholds = [
        threshold_at_fpr(full_plus[repeat][benign], 0.01) for repeat in range(repeats)
    ]
    ref_tpr = [
        _macro_tpr(full_plus[repeat], labels, threshold)
        for repeat, threshold in enumerate(ref_thresholds)
    ]
    payload["full_repo_plus"] = {
        "threshold_mean": float(np.mean(ref_thresholds)),
        "tpr_macro_mean": float(np.mean([item["macro"] for item in ref_tpr])),
        "tpr_weighted_mean": float(np.mean([item["weighted"] for item in ref_tpr])),
        "masks_used": int(n_masks),
    }

    cheap_full = {
        n_cheap: np.stack(
            [cheap_repo_plus(errors[repeat], n_cheap) for repeat in range(repeats)]
        )
        for n_cheap in cheap_options
    }

    for n_cheap in cheap_options:
        for fn_rate in fn_rates:
            key = f"cheap{n_cheap}_fn{fn_rate}"
            t_lows, thresholds = [], []
            tp, skip, fn, masks = [], [], [], []
            for repeat in range(repeats):
                t_low = cascade_threshold(
                    cheap_full[n_cheap][repeat][~benign], false_negative_rate=fn_rate
                )
                # Calibrate the final threshold on the cascade scores of the
                # benign samples, then apply the same rule to the attacks.
                probe = cascade_decisions(
                    errors[repeat], n_cheap=n_cheap, t_low=t_low, threshold=0.0
                )
                threshold = threshold_at_fpr(probe.score[benign], 0.01)
                result = cascade_decisions(
                    errors[repeat], n_cheap=n_cheap, t_low=t_low, threshold=threshold
                )
                t_lows.append(t_low)
                thresholds.append(threshold)
                tp.append(_macro_tpr(result.score, labels, threshold))
                skip.append(float((~result.routed[benign]).mean()))
                fn.append(float((~result.routed[~benign]).mean()))
                masks.append(_mix_masks(result.masks_used, labels))

            payload["cascade"][key] = {
                "n_cheap": int(n_cheap),
                "false_negative_rate": float(fn_rate),
                "t_low_mean": float(np.mean(t_lows)),
                "threshold_mean": float(np.mean(thresholds)),
                "tpr_macro_mean": float(np.mean([item["macro"] for item in tp])),
                "tpr_weighted_mean": float(np.mean([item["weighted"] for item in tp])),
                "benign_skip_rate": float(np.mean(skip)),
                "cheap_false_negative_rate": float(np.mean(fn)),
                "masks_used_mix": float(np.mean(masks)),
            }
            print(
                f"cascade cheap={n_cheap} fn={fn_rate}: "
                f"tpr={payload['cascade'][key]['tpr_macro_mean']:.3f} "
                f"skip={payload['cascade'][key]['benign_skip_rate']:.3f} "
                f"masks={payload['cascade'][key]['masks_used_mix']:.1f}"
            )

    # Sequential RePO (mean based).
    seq_thresholds = [
        threshold_at_fpr(full_repo[repeat][benign], 0.01) for repeat in range(repeats)
    ]
    seq_tpr, seq_masks, seq_agreement, seq_fpr = [], [], [], []
    for repeat, threshold in enumerate(seq_thresholds):
        result = sequential_repo_decisions(
            errors[repeat], threshold=threshold, block=5, min_masks=5
        )
        point = full_repo[repeat] >= threshold
        seq_tpr.append(_macro_tpr(full_repo[repeat], labels, threshold))
        seq_masks.append(_mix_masks(result.masks_used, labels))
        seq_agreement.append(float((result.decision == point).mean()))
        seq_fpr.append(float(result.decision[benign].mean()))
    payload["sequential_repo"] = {
        "threshold_mean": float(np.mean(seq_thresholds)),
        "tpr_macro_mean": float(np.mean([item["macro"] for item in seq_tpr])),
        "tpr_weighted_mean": float(np.mean([item["weighted"] for item in seq_tpr])),
        "masks_used_mix": float(np.mean(seq_masks)),
        "agreement_with_full": float(np.mean(seq_agreement)),
        "fpr_mean": float(np.mean(seq_fpr)),
    }
    print(
        f"sequential repo: tpr={payload['sequential_repo']['tpr_macro_mean']:.3f} "
        f"masks={payload['sequential_repo']['masks_used_mix']:.1f} "
        f"agree={payload['sequential_repo']['agreement_with_full']:.3f}"
    )

    payload["runtime_seconds"] = time.perf_counter() - started
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "flw_model"))
    parser.add_argument("--cache", default="")
    parser.add_argument("--max-benign", type=int, default=5_000)
    parser.add_argument("--max-per-attack", type=int, default=300)
    parser.add_argument("--n-masks", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cheap", default="5,10,25")
    parser.add_argument("--fn-rates", default="0.0,0.001,0.01")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "mask_efficiency" / "cascade_flow.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_cascade(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        cache=Path(args.cache) if args.cache else None,
        max_benign=args.max_benign,
        max_per_attack=args.max_per_attack,
        n_masks=args.n_masks,
        repeats=args.repeats,
        seed=args.seed,
        batch_size=args.batch_size,
        cheap_options=[int(value) for value in args.cheap.split(",")],
        fn_rates=[float(value) for value in args.fn_rates.split(",")],
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
