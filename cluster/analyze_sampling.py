#!/usr/bin/env python3
"""Analisa o viés de amostragem dos scores normais salvos pelo runner.

Os notebooks e o artigo avaliam as **primeiras N amostras** de cada ataque
(5.000 no fluxo, 25.000 nos pacotes). Este script compara esse recorte com o
conjunto completo e com N amostras sorteadas, para medir o quanto o critério
"primeiras N" enviesa o TPR.

Uso:
  python cluster/analyze_sampling.py --results-root "$RESULTS_ROOT" --model flow
  python cluster/analyze_sampling.py --results-root "$RESULTS_ROOT" --model packet

Lê ``<results-root>/<model>/normal/scores_<modo>.npz`` e imprime tabelas em
Markdown. Não importa TensorFlow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FLOW_NAMES = (
    "BENIGN", "FTP-Patator", "SSH-Patator", "DoS slowloris",
    "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye", "Heartbleed",
    "Web Attack", "Infiltration", "Bot", "PortScan", "DDoS",
)
PACKET_NAMES = (
    "Benign", "FTP-Patator", "SSH-Patator", "Slowloris", "Slowhttptest",
    "Hulk", "GoldenEye", "Heartbleed", "Web-Attack", "Infiltration",
    "Botnet", "PortScan", "DDoS",
)
DEFAULT_N_FIRST = {"flow": 5000, "packet": 25000}


def threshold_at_fpr(scores: np.ndarray, fpr: float) -> float:
    index = int(np.ceil(len(scores) * fpr))
    index = max(1, min(index, len(scores)))
    return float(np.sort(scores)[-index])


def mask_for(labels: np.ndarray, names: tuple[str, ...], name: str) -> np.ndarray:
    if labels.dtype.kind in "USO":
        return labels == name
    return labels == names.index(name)


def analyze_model(
    root: Path,
    model: str,
    n_first: int,
    draws: int,
    seed: int,
) -> None:
    names = FLOW_NAMES if model == "flow" else PACKET_NAMES
    rng = np.random.default_rng(seed)

    for mode in ("repo", "repo_plus"):
        path = root / model / "normal" / f"scores_{mode}.npz"
        if not path.is_file():
            print(f"[skip] {path} não existe")
            continue
        data = np.load(path, allow_pickle=True)
        scores = data["scores"].astype(np.float64)
        labels = data["labels"]
        threshold = float(data["threshold"])
        benign = scores[mask_for(labels, names, names[0])]

        print(f"\n### {model}/{mode} (n={len(scores)}, threshold={threshold:.6f})\n")
        print("| ataque | n | TPR completo | TPR primeiros N | TPR N aleatórios |")
        print("|---|---|---|---|---|")
        for name in names[1:]:
            mask = mask_for(labels, names, name)
            values = scores[mask]
            if len(values) == 0:
                continue
            full = float(np.mean(values >= threshold))
            first = float(np.mean(values[:n_first] >= threshold))
            sampled = []
            for _ in range(draws):
                if len(values) <= n_first:
                    sample = values
                else:
                    sample = rng.choice(values, size=n_first, replace=False)
                sampled.append(float(np.mean(sample >= threshold)))
            sampled = np.asarray(sampled)
            print(
                f"| {name} | {len(values)} | {full:.4f} | {first:.4f} | "
                f"{sampled.mean():.4f} ± {sampled.std():.4f} |"
            )

        print("\n| FPR alvo | threshold | FPR real | TPR médio dos ataques |")
        print("|---|---|---|---|")
        for target in (0.001, 0.005, 0.01, 0.05, 0.10):
            value = threshold_at_fpr(benign, target)
            observed = float(np.mean(benign >= value))
            rates = []
            for name in names[1:]:
                mask = mask_for(labels, names, name)
                if mask.any():
                    rates.append(float(np.mean(scores[mask] >= value)))
            print(f"| {target:.3f} | {value:.6f} | {observed:.4f} | {np.mean(rates):.4f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--model", choices=("flow", "packet"), required=True)
    parser.add_argument("--n-first", type=int, default=None)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2019)
    args = parser.parse_args()

    n_first = args.n_first or DEFAULT_N_FIRST[args.model]
    analyze_model(
        Path(args.results_root).expanduser().resolve(),
        args.model,
        n_first,
        args.draws,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
