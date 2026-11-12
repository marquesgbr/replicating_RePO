"""Merge per-attack JSON documents produced by the SLURM arrays.

The cluster runs one job per attack to parallelize the expensive adversarial
loop; each job writes its own JSON.  This script consolidates the pieces into
the single document consumed by the Markdown aggregator.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

from mask_experiments.runner import REPO_ROOT
from new_data_eval.results import write_json_atomic

__all__ = [
    "merge_adversarial",
    "merge_cascade_attack",
    "merge_distill_attack",
    "merge_variance",
    "main",
]

DEFAULT_RESULTS = REPO_ROOT / "results" / "mask_efficiency"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def merge_adversarial(directory: Path, out: Path) -> dict[str, Any]:
    """Merge the per-attack ``adversarial/<attack>.json`` files."""
    merged: dict[str, Any] | None = None
    for path in _files(directory):
        data = _load(path)
        if merged is None:
            merged = copy.deepcopy(data)
            merged["variants"] = {}
        for variant, item in data.get("variants", {}).items():
            target = merged["variants"].setdefault(
                variant,
                {
                    "policy": item["policy"],
                    "n_masks": item["n_masks"],
                    "real_threshold": item["real_threshold"],
                    "attack_threshold": item["attack_threshold"],
                    "attacks": {},
                    "cost": {"masked_rows": 0, "model_calls": 0, "model_seconds": 0.0},
                },
            )
            target["attacks"].update(item.get("attacks", {}))
            cost = item.get("cost", {})
            for key in ("masked_rows", "model_calls", "model_seconds"):
                target["cost"][key] += int(cost.get(key, 0)) if key != "model_seconds" else float(cost.get(key, 0.0))
    if merged is None:
        raise FileNotFoundError(f"no JSON files in {directory}")
    merged["dataset"]["attacks"] = {
        attack: result["n"]
        for variant in merged["variants"].values()
        for attack, result in variant["attacks"].items()
    }
    write_json_atomic(out, merged)
    return merged


def merge_cascade_attack(directory: Path, out: Path) -> dict[str, Any]:
    """Merge the per-attack ``cascade_attack/<attack>.json`` files."""
    merged: dict[str, Any] | None = None
    for path in _files(directory):
        data = _load(path)
        if merged is None:
            merged = copy.deepcopy(data)
            merged["attacks"] = {}
        merged["configs"].update(data.get("configs", {}))
        merged["attacks"].update(data.get("attacks", {}))
    if merged is None:
        raise FileNotFoundError(f"no JSON files in {directory}")
    merged["dataset"]["attacks"] = {
        attack: next(iter(configs.values()))["n"]
        for attack, configs in merged["attacks"].items()
    }
    write_json_atomic(out, merged)
    return merged


def merge_distill_attack(directory: Path, out: Path) -> dict[str, Any]:
    """Merge the per-attack ``distill_attack/<attack>.json`` files."""
    merged: dict[str, Any] | None = None
    for path in _files(directory):
        data = _load(path)
        if merged is None:
            merged = copy.deepcopy(data)
            merged["attacks"] = {}
        merged["attacks"].update(data.get("attacks", {}))
    if merged is None:
        raise FileNotFoundError(f"no JSON files in {directory}")
    merged["dataset"]["attacks"] = {
        attack: result["n"] for attack, result in merged["attacks"].items()
    }
    write_json_atomic(out, merged)
    return merged


def merge_variance(root: Path, out: Path | None = None) -> dict[str, Any]:
    """Combine the per-policy ``variance_flow_<policy>.json`` partials.

    The curve is split across SLURM jobs (one per mask policy) so each run
    stays inside the queue walltime.  Every partial scores the same sample
    with the same mask counts and repeats, so only ``policies`` may differ.
    """
    parts = sorted(root.glob("variance_flow_*.json"))
    if not parts:
        raise FileNotFoundError(f"no variance_flow_*.json in {root}")
    merged: dict[str, Any] | None = None
    for path in parts:
        data = _load(path)
        if merged is None:
            merged = copy.deepcopy(data)
            merged["results"] = {}
            merged["runtime_seconds"] = 0.0
        else:
            if data.get("dataset") != merged.get("dataset"):
                raise ValueError(f"dataset differs in {path}")
            left = dict(data.get("scoring", {}))
            right = dict(merged.get("scoring", {}))
            left.pop("policies", None)
            right.pop("policies", None)
            if left != right:
                raise ValueError(f"scoring differs in {path}")
        for policy, item in data.get("results", {}).items():
            if policy in merged["results"]:
                raise ValueError(f"policy {policy} duplicated in {path}")
            merged["results"][policy] = item
        merged["runtime_seconds"] = float(merged["runtime_seconds"]) + float(
            data.get("runtime_seconds", 0.0)
        )
    merged["scoring"]["policies"] = list(merged["results"])
    out = root / "variance_flow.json" if out is None else out
    write_json_atomic(out, merged)
    return merged


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.results_dir)
    try:
        merged = merge_variance(root)
    except FileNotFoundError:
        pass
    else:
        print(f"merged variance partials -> {root / 'variance_flow.json'}")
        print(f"policies: {sorted(merged['results'])}")
    for name, merger, out_name in (
        ("adversarial", merge_adversarial, "adversarial_all.json"),
        ("cascade_attack", merge_cascade_attack, "cascade_attack_all.json"),
        ("distill_attack", merge_distill_attack, "distill_attack_all.json"),
    ):
        directory = root / name
        if not directory.is_dir():
            continue
        out = root / out_name
        merger(directory, out)
        print(f"merged {directory} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
