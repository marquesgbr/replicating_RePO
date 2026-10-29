#!/usr/bin/env python3
"""Agrega os resultados JSON do runner e gera CSV/Markdown legíveis.

Exemplo:
  python cluster/aggregate_results.py \
    --results-root "$RESULTS_ROOT" \
    --out results/cluster
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CLUSTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLUSTER_DIR.parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import metrics  # noqa: E402

SUMMARY_FIELDS = [
    "model_type",
    "experiment",
    "score_mode",
    "attack_type",
    "threshold",
    "threshold_fpr",
    "fpr",
    "paper_threshold",
    "fpr_at_paper_threshold",
    "normal_tpr",
    "victim_tpr",
    "attack_success_rate",
    "fooled_rate",
    "average_tpr",
    "paper_average_tpr",
    "n_samples",
    "runtime_seconds",
    "run_id",
    "finished_at",
    "git_commit",
    "config_hash",
    "result_path",
]


def iter_result_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {"manifests", "training", "attack-lists", "logs"}:
            continue
        if path.name.endswith("training_manifest.json"):
            continue
        yield path


def load_results(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in iter_result_files(root):
        try:
            payload = metrics.read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[warn] ignorando JSON inválido {path}: {error}", file=sys.stderr)
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") != "completed":
            continue
        if not payload.get("experiment") or not payload.get("model_type"):
            continue
        payload["_path"] = str(path)
        results.append(payload)
    return results


def deduplicate(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        key = (
            str(result.get("model_type")),
            str(result.get("experiment")),
            str(result.get("score_mode")),
            str(result.get("attack_type") or ""),
        )
        previous = latest.get(key)
        if previous is None or str(result.get("finished_at", "")) >= str(previous.get("finished_at", "")):
            latest[key] = result
    return sorted(
        latest.values(),
        key=lambda item: (
            str(item.get("model_type")),
            str(item.get("experiment")),
            str(item.get("score_mode")),
            str(item.get("attack_type") or ""),
        ),
    )


def flatten(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result.get("config") or {}
    git = config.get("git") or result.get("git") or {}
    return {
        "model_type": result.get("model_type"),
        "experiment": result.get("experiment"),
        "score_mode": result.get("score_mode"),
        "attack_type": result.get("attack_type") or "",
        "threshold": result.get("threshold"),
        "threshold_fpr": result.get("threshold_fpr"),
        "fpr": result.get("fpr"),
        "paper_threshold": result.get("paper_threshold"),
        "fpr_at_paper_threshold": result.get("fpr_at_paper_threshold"),
        "normal_tpr": result.get("normal_tpr"),
        "victim_tpr": result.get("victim_tpr"),
        "attack_success_rate": result.get("attack_success_rate"),
        "fooled_rate": result.get("fooled_rate"),
        "average_tpr": result.get("average_tpr"),
        "paper_average_tpr": result.get("paper_average_tpr") or metrics.PAPER_AVERAGES.get(
            (
                str(result.get("model_type")),
                str(result.get("experiment")),
                str(result.get("score_mode")),
            )
        ),
        "n_samples": result.get("n_samples"),
        "runtime_seconds": result.get("runtime_seconds"),
        "run_id": result.get("run_id"),
        "finished_at": result.get("finished_at"),
        "git_commit": git.get("commit"),
        "config_hash": result.get("config_hash"),
        "result_path": result.get("_path"),
        # Mantido para o Markdown; não entra no CSV.
        "class_rates": result.get("class_rates"),
    }


def missing_results(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {
        (
            str(row.get("model_type")),
            str(row.get("experiment")),
            str(row.get("score_mode")),
            str(row.get("attack_type") or ""),
        )
        for row in rows
    }
    missing: list[str] = []
    for model_type in ("flow", "packet"):
        for mode in ("repo", "repo_plus"):
            key = (model_type, "normal", mode, "")
            if key not in present:
                missing.append(f"normal {model_type}/{mode}")
        for attack in metrics.expected_attacks(model_type):
            key = (model_type, "adversarial", "repo_plus", attack)
            if key not in present:
                missing.append(f"adversarial {model_type}/{attack}")
    return missing


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> None:
    lines: list[str] = ["# Resultados da replicação no Apuana", ""]
    lines.append("Gerado por `cluster/aggregate_results.py`.")
    lines.append("")

    normal_rows = [row for row in rows if row.get("experiment") == "normal"]
    if normal_rows:
        lines.append("## Avaliação normal (FPR alvo = 0,01)")
        lines.append("")
        for model_type in ("flow", "packet"):
            model_rows = [row for row in normal_rows if row.get("model_type") == model_type]
            if not model_rows:
                continue
            lines.append(f"### Modelo: {model_type}")
            lines.append("")
            lines.append("| Ataque | " + " | ".join(f"TPR {row['score_mode']}" for row in model_rows) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in model_rows) + " |")
            attacks = sorted(
                {
                    attack
                    for row in model_rows
                    for attack in (row.get("class_rates") or {}).keys()
                    if attack not in {"BENIGN", "Benign", "Web Attack", "Web-Attack"}
                }
            )
            for attack in attacks:
                cells = []
                for row in model_rows:
                    rate = ((row.get("class_rates") or {}).get(attack) or {}).get("rate")
                    cells.append(fmt(rate))
                lines.append(f"| {attack} | " + " | ".join(cells) + " |")
            lines.append("")
            for row in model_rows:
                lines.append(
                    f"- **{row['score_mode']}**: threshold={fmt(row.get('threshold'), 6)}, "
                    f"FPR={fmt(row.get('fpr'))}, FPR@paper={fmt(row.get('fpr_at_paper_threshold'))}, "
                    f"média TPR={fmt(row.get('average_tpr'))}, "
                    f"média do artigo={fmt(row.get('paper_average_tpr'))}"
                )
            lines.append("")

    adversarial_rows = [row for row in rows if row.get("experiment") == "adversarial"]
    if adversarial_rows:
        lines.append("## Avaliação adversarial (FPR de referência = 0,10)")
        lines.append("")
        lines.append("| Modelo | Ataque | TPR normal | TPR vítima | Sucesso do ataque | Fooled | Runtime (h) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in adversarial_rows:
            runtime = row.get("runtime_seconds")
            runtime_h = f"{float(runtime) / 3600:.2f}" if runtime is not None else "—"
            lines.append(
                "| {model} | {attack} | {normal} | {victim} | {success} | {fooled} | {runtime} |".format(
                    model=row.get("model_type"),
                    attack=row.get("attack_type"),
                    normal=fmt(row.get("normal_tpr")),
                    victim=fmt(row.get("victim_tpr")),
                    success=fmt(row.get("attack_success_rate")),
                    fooled=fmt(row.get("fooled_rate")),
                    runtime=runtime_h,
                )
            )
        lines.append("")

    if missing:
        lines.append("## Resultados ausentes")
        lines.append("")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results" / "cluster"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "cluster"))
    parser.add_argument("--strict", action="store_true", help="falha se faltar algum resultado esperado")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results(results_root)
    rows = [flatten(result) for result in deduplicate(raw)]
    missing = missing_results(rows)

    write_csv(out_dir / "summary.csv", rows)
    write_markdown(out_dir / "summary.md", rows, missing)
    metrics.write_json_atomic(
        out_dir / "summary.json",
        {
            "generated_at": metrics.iso_now(),
            "results_root": str(results_root),
            "n_results": len(rows),
            "missing": missing,
            "results": rows,
        },
    )

    print(f"[ok] {len(rows)} resultados agregados em {out_dir}")
    if missing:
        print(f"[warn] {len(missing)} resultados ausentes (veja summary.md)")
    if args.strict and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
