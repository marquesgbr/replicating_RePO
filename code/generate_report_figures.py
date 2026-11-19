"""Gera figuras (PDF) para o relatório (report.tex)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_replicating_repo"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def generate_cicids_adversarial_figure() -> Path:
    """Figura 1: TPR Normal vs. TPR Vítima (FPR = 0,10) no CICIDS2017 (Fluxos e Pacotes)."""
    attacks_flow = [
        "DoS Hulk",
        "Slowhttptest",
        "DoS GoldenEye",
        "Infiltration*",
        "DoS Slowloris",
        "DDoS",
        "FTP-Patator",
        "PortScan",
        "Bot",
        "SSH-Patator",
        "Heartbleed*",
    ]
    flow_normal = [0.9860, 0.9048, 0.7454, 0.6944, 0.6824, 0.5746, 0.4994, 0.1128, 0.0302, 0.0014, 1.0000]
    flow_victim = [0.9432, 0.8708, 0.6718, 0.6389, 0.6202, 0.5836, 0.0920, 0.1174, 0.0261, 0.0010, 1.0000]

    attacks_pkt = [
        "PortScan",
        "Hulk",
        "Slowhttptest",
        "Slowloris",
        "FTP-Patator",
        "Botnet",
        "DDoS",
        "GoldenEye",
        "Infiltration",
        "SSH-Patator",
        "Heartbleed",
    ]
    pkt_normal = [0.9913, 0.9832, 0.9149, 0.8822, 0.7451, 0.6653, 0.6542, 0.5785, 0.2360, 0.1730, 0.0261]
    pkt_victim = [0.9852, 0.9719, 0.9096, 0.8725, 0.7157, 0.6438, 0.6003, 0.5435, 0.2354, 0.1436, 0.0001]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), sharey=True)
    width = 0.38

    # Painel (a): Fluxos
    x_f = np.arange(len(attacks_flow))
    ax = axes[0]
    ax.bar(x_f - width / 2, flow_normal, width, label="TPR Normal (antes)", color="#2b5c8f", edgecolor="none")
    ax.bar(x_f + width / 2, flow_victim, width, label="TPR Vítima (após)", color="#d95f02", edgecolor="none")
    ax.axhline(0.10, color="#b22222", linestyle="--", linewidth=0.9, label="Limiar FPR = 0,10")
    ax.set_title("(a) Fluxos — CICIDS2017 (8/11 > FPR)", fontsize=8.8)
    ax.set_ylabel("Taxa de Detecção (TPR)")
    ax.set_xticks(x_f)
    ax.set_xticklabels(attacks_flow, rotation=38, ha="right")
    ax.set_ylim(0, 1.08)
    ax.annotate(
        "-40,7 p.p.",
        xy=(6 + width / 2, 0.105),
        xytext=(6.65, 0.42),
        fontsize=6.8,
        fontweight="bold",
        color="#8c2d04",
        arrowprops={"arrowstyle": "->", "color": "#8c2d04", "lw": 0.75},
    )

    # Painel (b): Pacotes
    x_p = np.arange(len(attacks_pkt))
    ax2 = axes[1]
    ax2.bar(x_p - width / 2, pkt_normal, width, label="TPR Normal (antes)", color="#2b5c8f", edgecolor="none")
    ax2.bar(x_p + width / 2, pkt_victim, width, label="TPR Vítima (após)", color="#d95f02", edgecolor="none")
    ax2.axhline(0.10, color="#b22222", linestyle="--", linewidth=0.9, label="Limiar FPR = 0,10")
    ax2.set_title("(b) Pacotes — CICIDS2017 (10/11 > FPR)", fontsize=8.8)
    ax2.set_xticks(x_p)
    ax2.set_xticklabels(attacks_pkt, rotation=38, ha="right")
    ax2.set_ylim(0, 1.08)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    out_path = IMAGES_DIR / "fig_cicids_adversarial.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_new_dataset_and_sampling_figure() -> Path:
    """Figura 2: (a) CIC-ToN-IoT (RePO vs RePO+ Normal vs RePO+ Vítima) e (b) Viés Temporal no CICIDS2017."""
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), gridspec_kw={"width_ratios": [1.32, 1.0]})

    # Painel (a): CIC-ToN-IoT @ FPR = 0,10
    classes_iot = [
        "backdoor",
        "password",
        "injection",
        "ransomware",
        "xss",
        "scanning",
        "mitm*",
        "ddos*",
        "dos*",
    ]
    repo_10 = [0.9945, 0.6757, 0.5890, 0.7421, 0.4554, 0.5101, 0.4371, 0.0050, 0.0138]
    repo_plus_10 = [0.9957, 0.9697, 0.9221, 0.9176, 0.7171, 0.6865, 0.5010, 0.0050, 0.0000]
    victim_10 = [0.9966, 0.9690, 0.9220, 0.9186, 0.7180, 0.6954, 0.4990, 0.0000, 0.0000]

    x_iot = np.arange(len(classes_iot))
    w = 0.27
    ax = axes[0]
    ax.bar(x_iot - w, repo_10, w, label="RePO (Normal)", color="#7b9acc")
    ax.bar(x_iot, repo_plus_10, w, label="RePO+ (Normal)", color="#1f4e78")
    ax.bar(x_iot + w, victim_10, w, label="RePO+ (Vítima Adv.)", color="#e07a2b")
    ax.axhline(0.10, color="#b22222", linestyle="--", linewidth=0.9, label="FPR = 0,10")
    ax.set_title("(a) CIC-ToN-IoT: Ganho RePO+ e Robustez", fontsize=8.8)
    ax.set_ylabel("Taxa de Detecção (TPR)")
    ax.set_xticks(x_iot)
    ax.set_xticklabels(classes_iot, rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=6.8)

    # Painel (b): Viés de amostragem temporal (CICIDS2017 Fluxo @ FPR = 0,01)
    sampled_attacks = ["DoS Hulk", "DDoS", "GoldenEye", "Slowloris", "Slowhttptest"]
    tpr_full = [0.6239, 0.3974, 0.2230, 0.1938, 0.1020]
    tpr_first5k = [0.6220, 0.2890, 0.2376, 0.2246, 0.1122]
    tpr_rand5k = [0.6238, 0.3965, 0.2223, 0.1942, 0.1023]
    tpr_rand_std = [0.0043, 0.0070, 0.0042, 0.0021, 0.0013]

    x_s = np.arange(len(sampled_attacks))
    w2 = 0.26
    ax2 = axes[1]
    ax2.bar(x_s - w2, tpr_full, w2, label="Completo", color="#2b5c8f")
    ax2.bar(x_s, tpr_first5k, w2, label="Primeiras 5.000", color="#c0392b")
    ax2.bar(
        x_s + w2,
        tpr_rand5k,
        w2,
        yerr=tpr_rand_std,
        capsize=2.0,
        label="5.000 Aleatórias",
        color="#27ae60",
    )
    ax2.set_title("(b) Viés de Amostragem (RePO+, FPR = 0,01)", fontsize=8.8)
    ax2.set_xticks(x_s)
    ax2.set_xticklabels(sampled_attacks, rotation=30, ha="right")
    ax2.set_ylim(0, 0.72)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=6.8)
    ax2.annotate(
        "-10,8 p.p.",
        xy=(1, 0.295),
        xytext=(1.15, 0.46),
        fontsize=6.8,
        fontweight="bold",
        color="#900c3f",
        arrowprops={"arrowstyle": "->", "color": "#900c3f", "lw": 0.75},
    )

    fig.tight_layout()
    out_path = IMAGES_DIR / "fig_new_dataset_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    p1 = generate_cicids_adversarial_figure()
    p2 = generate_new_dataset_and_sampling_figure()
    print(f"Generated: {p1}")
    print(f"Generated: {p2}")


if __name__ == "__main__":
    main()
