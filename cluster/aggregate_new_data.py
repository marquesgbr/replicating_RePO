#!/usr/bin/env python3
"""Consolidate the new dataset results into CSV, Markdown and JSON.

The layout produced by ``python -m new_data_eval`` is:

``<results>/normal/repo.json``
``<results>/normal/repo_plus.json``
``<results>/adversarial/<variant>/<attack>.json``
``<results>/smoke/...`` (ignored)

Only the standard library is used, so this runs on the login node without
TensorFlow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NORMAL_MODES = ("repo", "repo_plus")
ADVERSARIAL_VARIANTS = (
    "primary",
    "sensitivity_first_n",
    "sensitivity_paper_0.8",
)
EXPECTED_ATTACKS = (
    "xss",
    "password",
    "injection",
    "scanning",
    "backdoor",
    "ransomware",
    "mitm",
    "ddos",
    "dos",
)

COLUMNS = (
    "experiment",
    "variant",
    "score_mode",
    "attack",
    "n",
    "threshold_fpr_0.01",
    "threshold_fpr_0.10",
    "fpr",
    "macro_tpr",
    "weighted_tpr",
    "macro_tpr_fpr_0.01",
    "weighted_tpr_fpr_0.01",
    "threshold_policy",
    "attack_threshold",
    "fpr_at_attack_threshold",
    "normal_tpr",
    "victim_tpr",
    "attack_success_rate",
    "conditional_drop",
    "fooled",
    "evaded",
    "backup",
    "cons_as_mal",
    "cons_as_ben",
    "fooled_rate",
    "runtime_seconds",
    "config_hash",
    "git_commit",
    "path",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_normal(results_dir: Path) -> list[dict[str, Any]]:
    """Return one summary row per normal result document."""
    rows: list[dict[str, Any]] = []
    for mode in NORMAL_MODES:
        path = results_dir / "normal" / f"{mode}.json"
        payload = _read_json(path)
        if payload is None:
            continue
        thresholds = payload.get("thresholds")
        thresholds = thresholds if isinstance(thresholds, dict) else {}
        fpr_payload = payload.get("fpr")
        fpr_payload = fpr_payload if isinstance(fpr_payload, dict) else {}
        averages = payload.get("average_tpr")
        averages = averages if isinstance(averages, dict) else {}
        at_ten = averages.get("at_fpr_0.10")
        at_ten = at_ten if isinstance(at_ten, dict) else {}
        at_one = averages.get("at_fpr_0.01")
        at_one = at_one if isinstance(at_one, dict) else {}
        rows.append(
            {
                "experiment": "normal",
                "variant": "",
                "score_mode": payload.get("score_mode") or mode,
                "attack": "",
                "n": _integer((payload.get("dataset") or {}).get("n_samples")),
                "threshold_fpr_0.01": _number(thresholds.get("fpr_0.01")),
                "threshold_fpr_0.10": _number(thresholds.get("fpr_0.10")),
                "fpr": _number(fpr_payload.get("at_fpr_0.10_threshold")),
                "macro_tpr": _number(at_ten.get("macro")),
                "weighted_tpr": _number(at_ten.get("weighted")),
                "fpr_at_attack_threshold": _number(
                    fpr_payload.get("at_fpr_0.10_threshold")
                ),
                "runtime_seconds": _number(payload.get("runtime_seconds")),
                "config_hash": payload.get("config_hash"),
                "git_commit": payload.get("git_commit"),
                "path": str(path),
            }
        )
        # Keep the 1% macro/weighted values available for callers that want
        # both operating points without parsing the JSON again.
        rows[-1]["macro_tpr_fpr_0.01"] = _number(at_one.get("macro"))
        rows[-1]["weighted_tpr_fpr_0.01"] = _number(at_one.get("weighted"))
    return rows


def load_adversarial(results_dir: Path) -> list[dict[str, Any]]:
    """Return one summary row per adversarial result document."""
    rows: list[dict[str, Any]] = []
    for variant in ADVERSARIAL_VARIANTS:
        directory = results_dir / "adversarial" / variant
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            if payload is None:
                continue
            thresholds = payload.get("thresholds")
            thresholds = thresholds if isinstance(thresholds, dict) else {}
            counts = payload.get("counts")
            counts = counts if isinstance(counts, dict) else {}
            rows.append(
                {
                    "experiment": "adversarial",
                    "variant": variant,
                    "score_mode": payload.get("score_mode"),
                    "attack": payload.get("attack_type")
                    or payload.get("attack")
                    or path.stem,
                    "n": _integer(counts.get("n")),
                    "threshold_fpr_0.10": _number(thresholds.get("fpr_0.10")),
                    "threshold_policy": thresholds.get("policy"),
                    "attack_threshold": _number(thresholds.get("attack")),
                    "fpr_at_attack_threshold": _number(
                        payload.get("fpr_at_attack_threshold")
                    ),
                    "normal_tpr": _number(
                        payload.get("normal_tpr_at_attack_threshold")
                    ),
                    "victim_tpr": _number(payload.get("victim_tpr")),
                    "attack_success_rate": _number(
                        payload.get("attack_success_rate")
                    ),
                    "conditional_drop": _number(payload.get("conditional_drop")),
                    "fooled": _integer(counts.get("fooled")),
                    "evaded": _integer(counts.get("evaded")),
                    "backup": _integer(counts.get("backup")),
                    "cons_as_mal": _integer(counts.get("cons_as_mal")),
                    "cons_as_ben": _integer(counts.get("cons_as_ben")),
                    "fooled_rate": _number(payload.get("fooled_rate")),
                    "runtime_seconds": _number(payload.get("runtime_seconds")),
                    "config_hash": payload.get("config_hash"),
                    "git_commit": payload.get("git_commit"),
                    "path": str(path),
                }
            )
    return rows


def missing_results(results_dir: Path) -> list[str]:
    """Return the expected result files that are absent."""
    missing: list[str] = []
    for mode in NORMAL_MODES:
        path = results_dir / "normal" / f"{mode}.json"
        if not path.is_file():
            missing.append(str(path.relative_to(results_dir)))
    for variant in ADVERSARIAL_VARIANTS:
        for attack in EXPECTED_ATTACKS:
            path = results_dir / "adversarial" / variant / f"{attack}.json"
            if not path.is_file():
                missing.append(str(path.relative_to(results_dir)))
    return missing


def summary_rows(results_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the combined rows and the missing result paths."""
    rows = load_normal(results_dir) + load_adversarial(results_dir)
    return rows, missing_results(results_dir)


def _format(value: Any, digits: int = 4) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _markdown(rows: list[dict[str, Any]], missing: list[str], results_dir: Path) -> str:
    lines = [
        "# Resumo do novo dataset",
        "",
        f"Resultados: `{results_dir}`",
        "",
        "## Normal",
        "",
        "| modo | n | threshold@1% | threshold@10% | FPR@10% | macro@1% | weighted@1% | macro@10% | weighted@10% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["experiment"] != "normal":
            continue
        lines.append(
            "| {mode} | {n} | {t1} | {t10} | {fpr} | {macro1} | {weighted1} | {macro} | {weighted} |".format(
                mode=row["score_mode"],
                n=row["n"] if row["n"] is not None else "-",
                t1=_format(row.get("threshold_fpr_0.01")),
                t10=_format(row.get("threshold_fpr_0.10")),
                fpr=_format(row.get("fpr")),
                macro1=_format(row.get("macro_tpr_fpr_0.01")),
                weighted1=_format(row.get("weighted_tpr_fpr_0.01")),
                macro=_format(row.get("macro_tpr")),
                weighted=_format(row.get("weighted_tpr")),
            )
        )

    for variant in ADVERSARIAL_VARIANTS:
        variant_rows = [
            row
            for row in rows
            if row["experiment"] == "adversarial" and row["variant"] == variant
        ]
        if not variant_rows:
            continue
        lines.extend(
            [
                "",
                f"## Adversarial — {variant}",
                "",
                "| ataque | n | victim_tpr | conditional_drop | fooled | FPR@ataque |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in variant_rows:
            lines.append(
                "| {attack} | {n} | {victim} | {drop} | {fooled} | {fpr} |".format(
                    attack=row["attack"],
                    n=row["n"] if row["n"] is not None else "-",
                    victim=_format(row.get("victim_tpr")),
                    drop=_format(row.get("conditional_drop")),
                    fooled=row.get("fooled", "-"),
                    fpr=_format(row.get("fpr_at_attack_threshold")),
                )
            )

    lines.extend(["", "## Resultados ausentes", ""])
    if missing:
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("Nenhum resultado ausente.")
    lines.append("")
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_summary(results_dir: Path, out_dir: Path) -> None:
    """Write ``summary.csv``, ``summary.md`` and ``summary.json``."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    rows, missing = summary_rows(results_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    descriptor, temporary = tempfile.mkstemp(
        prefix=".summary.", suffix=".csv", dir=str(out_dir)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=COLUMNS, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temporary, out_dir / "summary.csv")
        os.chmod(out_dir / "summary.csv", 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    _write_atomic(out_dir / "summary.md", _markdown(rows, missing, results_dir))

    payload = {
        "results_dir": str(results_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": rows,
        "missing": missing,
    }
    _write_atomic(out_dir / "summary.json", json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the new dataset normal/adversarial results."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="directory containing normal/ and adversarial/",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="directory that receives summary.csv, summary.md and summary.json",
    )
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"ERROR: results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    write_summary(args.results_dir, args.out)
    rows, missing = summary_rows(args.results_dir)
    print(f"[ok] {len(rows)} resultados, {len(missing)} ausentes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
