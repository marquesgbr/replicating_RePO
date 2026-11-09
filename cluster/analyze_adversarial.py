#!/usr/bin/env python3
"""Consolida os resultados adversariais de um modelo.

Além da tabela por ataque, calcula as três formas de média usadas no relatório:

* macro sobre todos os ataques;
* macro apenas nos ataques com n >= ``--min-samples``;
* ponderada por amostras.

Também calcula a queda condicional entre as amostras originalmente detectadas,
que é mais comparável entre ataques do que ``1 - victim_tpr``.

Uso:
  python cluster/analyze_adversarial.py --results-root "$RESULTS_ROOT" --model flow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FPR = 0.10


def load_results(root: Path, model: str) -> list[dict[str, Any]]:
    directory = root / model / "adversarial"
    results = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            results.append(payload)
    return results


def conditional_drop(normal: float, victim: float) -> float | None:
    """Queda relativa entre as amostras que o modelo detectava antes."""
    if normal <= 0:
        return None
    return (normal - victim) / normal


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--model", choices=("flow", "packet"), required=True)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--fpr", type=float, default=FPR)
    args = parser.parse_args()

    results = load_results(Path(args.results_root).expanduser().resolve(), args.model)
    if not results:
        print("nenhum resultado adversarial encontrado")
        return 1

    print("| Ataque | n | TPR normal | TPR vítima | Queda (pp) | Queda condicional | Fooled | Tempo (min) |")
    print("|---|---|---|---|---|---|---|---|")
    for item in results:
        normal = float(item.get("normal_tpr") or 0.0)
        victim = float(item.get("victim_tpr") or 0.0)
        drop = conditional_drop(normal, victim)
        print(
            "| {attack} | {n} | {normal} | {victim} | {delta:+.4f} | {drop} | {fooled} | {runtime:.1f} |".format(
                attack=item.get("attack_type"),
                n=item.get("n_samples", 0),
                normal=fmt(normal),
                victim=fmt(victim),
                delta=victim - normal,
                drop=fmt(drop),
                fooled=fmt(item.get("fooled_rate")),
                runtime=(item.get("runtime_seconds") or 0) / 60,
            )
        )

    total_n = sum(int(item.get("n_samples", 0)) for item in results)
    macro_all = sum(float(item.get("victim_tpr") or 0) for item in results) / len(results)
    large = [item for item in results if int(item.get("n_samples", 0)) >= args.min_samples]
    macro_large = (
        sum(float(item.get("victim_tpr") or 0) for item in large) / len(large)
        if large
        else None
    )
    weighted = (
        sum(float(item.get("victim_tpr") or 0) * int(item.get("n_samples", 0)) for item in results)
        / total_n
        if total_n
        else None
    )
    above = sum(1 for item in results if float(item.get("victim_tpr") or 0) > args.fpr)

    print()
    print(f"- ataques: {len(results)}")
    print(f"- amostras: {total_n}")
    print(f"- média macro (todos): {macro_all:.4f}")
    print(f"- média macro (n >= {args.min_samples}): {fmt(macro_large)}")
    print(f"- média ponderada por amostras: {fmt(weighted)}")
    print(f"- ataques acima do FPR {args.fpr:.2f}: {above} de {len(results)}")
    print(f"- tempo de GPU somado: {sum(float(i.get('runtime_seconds') or 0) for i in results) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
