"""Normal-evaluation pilot on the packet model (Task 5).

Only the normal evaluation is run: the packet attack is far more expensive than
the flow attack (hundreds of iterations per packet) and the point of this pilot
is to check whether the mask-policy conclusions transfer to the 2-D packet
input.

The packet windows are ``(20, 29)``; the mask policy is applied to the
flattened 580 cells and reshaped back, so the keep rule and the stateless seeds
are exactly the ones of the flow study.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mask_experiments.runner import REPO_ROOT, load_flow_model
from new_data_eval.masking import POLICIES, MaskPolicy
from new_data_eval.results import write_json_atomic
from new_data_eval.scoring import threshold_at_fpr

__all__ = ["PACKET_LABELS", "run_packet"]

TIMESTEPS = 20
N_FEATURES = 29
WINDOW_FEATURES = TIMESTEPS * N_FEATURES

PACKET_LABELS: tuple[str, ...] = (
    "Benign",
    "FTP-Patator",
    "SSH-Patator",
    "Slowloris",
    "Slowhttptest",
    "Hulk",
    "GoldenEye",
    "Heartbleed",
    "Web-Attack",
    "Infiltration",
    "Botnet",
    "PortScan",
    "DDoS",
)

#: Packet attacks averaged in the paper (Web-Attack excluded).
PACKET_ATTACKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)

DEFAULT_VARIANTS: tuple[tuple[str, int], ...] = (
    ("iid", 100),
    ("balanced", 25),
    ("covering", 25),
)


def _windows(
    day_dirs: Iterable[Path],
    *,
    windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a class-balanced ``(x, labels, indices)`` sample of windows.

    The packet days are huge and the attacks are concentrated in time, so a
    purely uniform draw can miss an attack class entirely.  The sampler draws
    at most ``windows / (days * n_classes)`` windows per class per day, which
    keeps the macro TPR comparable across classes.  The global index includes
    the day number, so two windows of different days never share masks.
    """
    dirs = [Path(item) for item in day_dirs]
    generator = np.random.default_rng(seed)
    per_day = max(1, windows // max(1, len(dirs)))

    x_parts: list[np.ndarray] = []
    label_parts: list[int] = []
    index_parts: list[int] = []

    for day_number, day_dir in enumerate(dirs):
        parts = sorted(glob.glob(str(day_dir / "part_*.npy")))
        if not parts:
            raise FileNotFoundError(f"part_*.npy ausentes em {day_dir}")
        labels = np.load(day_dir / "labels.npy", mmap_mode="r")
        classes = [
            int(value)
            for value in np.unique(np.asarray(labels[:], dtype=np.int64))
            if value >= 0
        ]
        quota = {name: max(1, per_day // max(1, len(classes))) for name in classes}
        collected: dict[int, list[tuple[np.ndarray, int]]] = {
            name: [] for name in classes
        }

        for part_number, part_path in enumerate(parts):
            part = np.load(part_path, mmap_mode="r")
            rows = part.shape[0]
            file_index = part_number * rows
            part_labels = np.asarray(
                labels[file_index : file_index + rows], dtype=np.int64
            )
            valid_starts = np.flatnonzero(part_labels[TIMESTEPS - 1 :] >= 0)
            if valid_starts.size == 0:
                continue
            start_labels = part_labels[valid_starts + TIMESTEPS - 1]
            for name in classes:
                remaining = quota[name] - len(collected[name])
                if remaining <= 0:
                    continue
                candidates = valid_starts[start_labels == name]
                if candidates.size == 0:
                    continue
                chosen = generator.choice(
                    candidates, size=min(remaining, candidates.size), replace=False
                )
                for start in np.sort(chosen):
                    collected[name].append(
                        (
                            np.asarray(
                                part[start : start + TIMESTEPS], dtype=np.float32
                            ),
                            day_number * 10_000_000 + file_index + int(start),
                        )
                    )

        for name in classes:
            for window, index in collected[name]:
                x_parts.append(window)
                label_parts.append(name)
                index_parts.append(index)

    if not x_parts:
        raise ValueError("no valid packet windows were sampled")
    return (
        np.stack(x_parts),
        np.asarray(label_parts, dtype=np.int64),
        np.asarray(index_parts, dtype=np.int64),
    )


def _score(
    model: Any,
    x_raw: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    *,
    policy: MaskPolicy,
    n_masks: int,
    seed: int,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the RePO and RePO+ scores of every window."""
    span = train_max - train_min + 1e-6
    x_norm = (x_raw - train_min) / span
    repo = np.empty(len(x_raw), dtype=np.float64)
    plus = np.empty(len(x_raw), dtype=np.float64)
    for row, sample_index in enumerate(indices):
        masks = policy.masks(
            seed=seed,
            sample_index=int(sample_index),
            phase="original",
            repeat=0,
            n_masks=n_masks,
            n_features=WINDOW_FEATURES,
        ).reshape(n_masks, TIMESTEPS, N_FEATURES)
        partial = x_norm[row] * masks
        reconstructed = model(partial, training=False)
        reconstructed = np.asarray(reconstructed, dtype=np.float64)
        errors = np.mean(
            (reconstructed - x_norm[row]) ** 2, axis=(1, 2)
        )
        repo[row] = errors.mean()
        plus[row] = errors.reshape(5, -1).min(axis=1).sum()
    return repo, plus


def _macro(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    tprs, counts = [], []
    for index in PACKET_ATTACKS:
        mask = labels == index
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


def run_packet(
    *,
    data_dir: Path,
    model_dir: Path,
    days: Iterable[str] = ("tuesday", "wednesday", "thursday", "friday"),
    windows: int = 600,
    seed: int = 2019,
    variants: Iterable[tuple[str, int]] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    started = time.perf_counter()
    day_list = [str(day) for day in days]
    day_dirs = [Path(data_dir) / "packet_based" / day for day in day_list]
    x_raw, labels, indices = _windows(day_dirs, windows=windows, seed=seed)
    meta_dir = Path(data_dir) / "packet_based" / "x_train_meta"
    train_min = np.load(meta_dir / "train_min.npy").astype(np.float32)
    train_max = np.load(meta_dir / "train_max.npy").astype(np.float32)
    model = load_flow_model(model_dir)

    payload: dict[str, Any] = {
        "schema_version": "mask_exp_v1",
        "experiment": "packet_normal",
        "dataset": {
            "name": "cicids2017-packet",
            "days": day_list,
            "n_windows": int(len(labels)),
            "seed": seed,
            "counts": {
                PACKET_LABELS[index]: int((labels == index).sum())
                for index in sorted(set(labels.tolist()))
            },
        },
        "variants": {},
        "runtime_seconds": 0.0,
    }

    benign = labels == 0
    for policy_name, n_masks in variants:
        policy = POLICIES[policy_name]
        repo, plus = _score(
            model,
            x_raw,
            train_min,
            train_max,
            policy=policy,
            n_masks=n_masks,
            seed=seed,
            indices=indices,
        )
        for mode, scores in (("repo", repo), ("repo_plus", plus)):
            key = f"{policy_name}-{n_masks}"
            payload["variants"].setdefault(
                key, {"policy": policy_name, "n_masks": int(n_masks)}
            )
            threshold = threshold_at_fpr(scores[benign], 0.01)
            payload["variants"][key][mode] = {
                "threshold_fpr_0.01": float(threshold),
                "fpr": float(np.mean(scores[benign] >= threshold)),
                "tpr": _macro(scores, labels, threshold),
            }
        print(
            f"{policy_name}-{n_masks}: "
            f"tpr={payload['variants'][f'{policy_name}-{n_masks}']['repo_plus']['tpr']['macro']:.3f}"
        )

    payload["runtime_seconds"] = time.perf_counter() - started
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "pkt_model"))
    parser.add_argument("--days", default="tuesday,wednesday,thursday,friday")
    parser.add_argument("--windows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--variants", default="iid:100,balanced:25,covering:25")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "mask_efficiency" / "packet_normal.json"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    variants = []
    for item in args.variants.split(","):
        policy, _, count = item.partition(":")
        variants.append((policy.strip(), int(count)))
    payload = run_packet(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        days=[item.strip() for item in args.days.split(",") if item.strip()],
        windows=args.windows,
        seed=args.seed,
        variants=variants,
    )
    write_json_atomic(Path(args.out), payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
