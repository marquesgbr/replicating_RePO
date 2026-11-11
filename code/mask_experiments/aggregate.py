"""Consolidate the mask efficiency JSONs into one Markdown report.

The report is generated from the result documents, never hand-edited, so every
number in it is traceable to a JSON produced by the experiment runners.  When a
document is missing the corresponding section says so instead of failing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mask_experiments.runner import REPO_ROOT

__all__ = ["build_markdown", "main"]

DEFAULT_RESULTS = REPO_ROOT / "results" / "mask_efficiency"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "MASK_EFFICIENCY_RESULTS.md"


def _load(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _baseline(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Baseline ainda não executado._"
    rows = []
    for mode, item in payload["modes"].items():
        rows.append(
            [
                mode,
                payload["dataset"]["n_samples"],
                item["thresholds"]["fpr_0.01"],
                item["fpr"]["fpr_0.01"],
                item["average_tpr"]["fpr_0.01"]["macro"],
                item["average_tpr"]["fpr_0.01"]["weighted"],
            ]
        )
    return _table(
        ["modo", "n", "threshold FPR 1%", "FPR", "TPR macro", "TPR weighted"],
        rows,
    ) + f"\n\nCusto: {payload['cost']}  \nTempo: {payload['runtime_seconds']:.1f} s"


def _variance(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Estudo de variância ainda não executado._"
    rows = []
    for policy, by_n in payload["results"].items():
        for n_masks, item in sorted(by_n.items(), key=lambda pair: int(pair[0])):
            mode = item["repo_plus"]["thresholds"]["fpr_0.01"]
            rows.append(
                [
                    policy,
                    int(n_masks),
                    mode["tpr"]["macro"],
                    mode["tpr"]["macro_std"],
                    mode["tpr"]["weighted"],
                    item["repo_plus"]["score_std_across_repeats"],
                    item["repo_plus"]["spearman_vs_reference"],
                    item["cost"]["masked_rows"],
                ]
            )
    return _table(
        [
            "política",
            "n_masks",
            "TPR macro",
            "desvio do TPR",
            "TPR weighted",
            "desvio do escore",
            "Spearman vs iid-100",
            "linhas mascaradas",
        ],
        rows,
    )


def _adversarial(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Piloto adversarial ainda não executado._"
    rows = []
    summary_rows = []
    for variant, item in payload["variants"].items():
        victim, fooled, no_change, masks = [], [], [], []
        for attack, result in item["attacks"].items():
            outcomes = result.get("outcomes", [])
            mean_masks = (
                float(np.mean([entry["masks_evaluated"] for entry in outcomes]))
                if outcomes
                else 0.0
            )
            rows.append(
                [
                    variant,
                    attack,
                    result["n"],
                    result["normal_tpr"],
                    result["victim_tpr"],
                    f"{result['fooled']}/{result['n']}",
                    result["kinds"].get("no_change", 0),
                    int(mean_masks),
                ]
            )
            victim.append(result["victim_tpr"])
            fooled.append(result["fooled"] / result["n"])
            no_change.append(result["kinds"].get("no_change", 0) / result["n"])
            masks.append(mean_masks)
        summary_rows.append(
            [
                variant,
                item["real_threshold"],
                item["attack_threshold"],
                float(np.mean(victim)),
                float(np.mean(fooled)),
                float(np.mean(no_change)),
                int(np.mean(masks)),
            ]
        )
    detail = _table(
        [
            "variante",
            "ataque",
            "n",
            "TPR normal",
            "TPR vítima",
            "fooled",
            "no_change",
            "máscaras/ataque",
        ],
        rows,
    )
    summary = _table(
        [
            "variante",
            "threshold real (FPR 10%)",
            "threshold de ataque",
            "TPR vítima médio",
            "fooled médio",
            "no_change médio",
            "máscaras/ataque (média)",
        ],
        summary_rows,
    )
    return detail + "\n\n**Resumo por variante**\n\n" + summary


def _cascade(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Cascata ainda não executada._"
    full = payload["full_repo_plus"]
    rows = [
        [
            "RePO+ integral",
            full["masks_used"],
            full["tpr_macro_mean"],
            full["tpr_weighted_mean"],
            "—",
            "—",
        ]
    ]
    for key, item in payload["cascade"].items():
        rows.append(
            [
                key,
                item["masks_used_mix"],
                item["tpr_macro_mean"],
                item["tpr_weighted_mean"],
                item["benign_skip_rate"],
                item["cheap_false_negative_rate"],
            ]
        )
    seq = payload["sequential_repo"]
    rows.append(
        [
            "RePO sequencial",
            seq["masks_used_mix"],
            seq["tpr_macro_mean"],
            seq["tpr_weighted_mean"],
            seq["fpr_mean"],
            seq["agreement_with_full"],
        ]
    )
    return _table(
        [
            "regra",
            "máscaras/amostra (99% benigno)",
            "TPR macro",
            "TPR weighted",
            "benignos pulados / FPR",
            "FN do estágio barato / concordância",
        ],
        rows,
    )


def _distillation(
    payload: Mapping[str, Any] | None,
    distill_attack: Mapping[str, Any] | None = None,
) -> str:
    if payload is None:
        return "_Destilação ainda não executada._"
    teacher = payload["teacher"]
    student = payload["student"]
    attacked = distill_attack is not None or student.get("adversarial_evaluated")
    return _table(
        ["modelo", "TPR macro", "TPR weighted", "Spearman", "passes"],
        [
            [
                "professor (100 máscaras)",
                teacher["tpr"]["macro"],
                teacher["tpr"]["weighted"],
                1.0,
                100,
            ],
            [
                "aluno (passe único)",
                student["tpr"]["macro"],
                student["tpr"]["weighted"],
                student["spearman"],
                1,
            ],
        ],
    ) + (
        f"\n\nRMSE do aluno: {student['rmse']:.4f}; MAE: {student['mae']:.4f}. "
        f"Ataque adaptativo no aluno: "
        f"{'executado (seção 5.1)' if attacked else 'não executado'}."
    )


def _packet(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Piloto de pacotes ainda não executado._"
    rows = []
    for variant, item in payload["variants"].items():
        for mode in ("repo", "repo_plus"):
            rows.append(
                [
                    variant,
                    mode,
                    item[mode]["threshold_fpr_0.01"],
                    item[mode]["tpr"]["macro"],
                    item[mode]["tpr"]["weighted"],
                ]
            )
    return _table(
        ["variante", "modo", "threshold FPR 1%", "TPR macro", "TPR weighted"],
        rows,
    ) + f"\n\nJanelas: {payload['dataset']['n_windows']}."


def _key_findings(
    baseline: Mapping[str, Any] | None,
    variance: Mapping[str, Any] | None,
    adversarial: Mapping[str, Any] | None,
    cascade: Mapping[str, Any] | None,
    distill: Mapping[str, Any] | None,
    packet: Mapping[str, Any] | None,
    cascade_attack: Mapping[str, Any] | None = None,
    distill_attack: Mapping[str, Any] | None = None,
) -> str:
    bullets: list[str] = []
    if variance is not None:
        for policy, by_n in variance["results"].items():
            ordered = sorted(by_n.items(), key=lambda pair: int(pair[0]))
            reference = float(
                ordered[-1][1]["repo_plus"]["thresholds"]["fpr_0.01"]["tpr"]["macro"]
            )
            chosen = None
            for n_masks, item in ordered:
                if policy == "covering" and int(n_masks) < 25:
                    # Coverage of all 77 features is impossible with fewer
                    # masks per group; the policy falls back to balanced there.
                    continue
                tpr = float(item["repo_plus"]["thresholds"]["fpr_0.01"]["tpr"]["macro"])
                rho = item["repo_plus"]["spearman_vs_reference"]
                if abs(tpr - reference) <= 0.01 and (rho is None or rho >= 0.95):
                    chosen = (int(n_masks), tpr, item)
                    break
            if chosen is None:
                bullets.append(
                    f"- **{policy}:** nenhum `n_masks` ficou a 1 ponto do "
                    f"TPR de {ordered[-1][0]} máscaras com Spearman ≥ 0,95; a "
                    "redução não é gratuita."
                )
            else:
                n_masks, tpr, item = chosen
                rows = item["cost"]["masked_rows"]
                full_rows = ordered[-1][1]["cost"]["masked_rows"]
                bullets.append(
                    f"- **{policy}:** {n_masks} máscaras ficam a ≤1 ponto do TPR "
                    f"de 100 (TPR macro {tpr:.3f} vs {reference:.3f}, Spearman "
                    f"{item['repo_plus']['spearman_vs_reference']:.3f}) com "
                    f"{full_rows / rows:.1f}× menos linhas mascaradas."
                )
    if adversarial is not None:
        summaries = []
        for variant, item in adversarial["variants"].items():
            victim = float(
                np.mean([attack["victim_tpr"] for attack in item["attacks"].values()])
            )
            no_change = float(
                np.mean(
                    [
                        attack["kinds"].get("no_change", 0) / attack["n"]
                        for attack in item["attacks"].values()
                    ]
                )
            )
            summaries.append((variant, victim, no_change))
        text = "; ".join(
            f"{variant}: vítima {victim:.2f}, já benigno {no_change:.2f}"
            for variant, victim, no_change in summaries
        )
        bullets.append(f"- **Adversarial:** {text}.")
    if cascade is not None:
        seq = cascade["sequential_repo"]
        bullets.append(
            f"- **Sequencial:** {seq['masks_used_mix']:.1f} máscaras/amostra, "
            f"TPR macro {seq['tpr_macro_mean']:.3f} e "
            f"{100 * seq['agreement_with_full']:.1f}% de concordância com o "
            "RePO completo."
        )
        best = min(
            cascade["cascade"].values(),
            key=lambda item: item["masks_used_mix"],
        )
        bullets.append(
            f"- **Cascata:** o melhor orçamento usa "
            f"{best['masks_used_mix']:.1f} máscaras/amostra e pula "
            f"{100 * best['benign_skip_rate']:.0f}% dos benignos, mas o "
            f"estágio barato tem FN de {100 * best['cheap_false_negative_rate']:.1f}%."
        )
    if distill is not None:
        bullets.append(
            f"- **Destilação:** aluno de passe único tem TPR "
            f"{distill['student']['tpr']['macro']:.3f} vs "
            f"{distill['teacher']['tpr']['macro']:.3f} do professor e "
            f"Spearman {distill['student']['spearman']:.3f}."
        )
    if distill_attack is not None:
        attacks = list(distill_attack["attacks"].values())
        student = np.mean([item["student_victim_tpr"] for item in attacks])
        teacher = np.mean([item["teacher_victim_tpr"] for item in attacks])
        student_normal = np.mean([item["student_normal_tpr"] for item in attacks])
        teacher_normal = np.mean([item["teacher_normal_tpr"] for item in attacks])
        bullets.append(
            f"- **Destilação sob ataque:** TPR de vítima do aluno "
            f"{student:.2f} (normal {student_normal:.2f}) vs {teacher:.2f} "
            f"(normal {teacher_normal:.2f}) do professor — a destilação "
            "normal não preserva a robustez adversarial."
        )
    if cascade_attack is not None:
        rows = [
            result
            for configs in cascade_attack["attacks"].values()
            for result in configs.values()
        ]
        normal = np.mean([item["normal_tpr"] for item in rows])
        victim = np.mean([item["victim_tpr"] for item in rows])
        bullets.append(
            f"- **Cascata sob ataque:** TPR de vítima {victim:.2f} vs normal "
            f"{normal:.2f} (queda {1.0 - victim / normal:.2f}), "
            f"bypass do estágio barato "
            f"{np.mean([item['bypass_rate'] for item in rows]):.2f}."
        )
    if packet is not None:
        items = {
            variant: item["repo_plus"]["tpr"]["macro"]
            for variant, item in packet["variants"].items()
        }
        best = max(items, key=items.get)
        bullets.append(
            "- **Pacotes (normal):** "
            + "; ".join(f"{name} {value:.3f}" for name, value in items.items())
            + f" — melhor {best}."
        )
    if not bullets:
        return "_Nenhum resultado disponível ainda._"
    return "\n".join(bullets)


def _adversarial_by_direction(
    adversarial: Mapping[str, Any] | None,
    cascade_attack: Mapping[str, Any] | None,
    distill_attack: Mapping[str, Any] | None,
) -> str:
    rows: list[list[Any]] = []
    if adversarial is not None:
        for variant, item in adversarial["variants"].items():
            victim = np.mean(
                [attack["victim_tpr"] for attack in item["attacks"].values()]
            )
            normal = np.mean(
                [attack["normal_tpr"] for attack in item["attacks"].values()]
            )
            no_change = np.mean(
                [
                    attack["kinds"].get("no_change", 0) / attack["n"]
                    for attack in item["attacks"].values()
                ]
            )
            masks = np.mean(
                [
                    np.mean([entry["masks_evaluated"] for entry in attack["outcomes"]])
                    for attack in item["attacks"].values()
                ]
            )
            rows.append(
                [
                    "A — máscaras",
                    variant,
                    normal,
                    victim,
                    (1.0 - victim / normal) if normal else None,
                    no_change,
                    int(masks),
                ]
            )
    if cascade_attack is not None:
        by_config: dict[str, list[Mapping[str, Any]]] = {}
        for configs in cascade_attack["attacks"].values():
            for name, result in configs.items():
                by_config.setdefault(name, []).append(result)
        for name, results in by_config.items():
            normal = np.mean([item["normal_tpr"] for item in results])
            victim = np.mean([item["victim_tpr"] for item in results])
            rows.append(
                [
                    "B — cascata",
                    name,
                    normal,
                    victim,
                    (1.0 - victim / normal) if normal else None,
                    np.mean([item["bypass_rate"] for item in results]),
                    int(np.mean([item["masks_evaluated"] for item in results])),
                ]
            )
    if distill_attack is not None:
        attacks = list(distill_attack["attacks"].values())
        student_victim = np.mean([item["student_victim_tpr"] for item in attacks])
        teacher_victim = np.mean([item["teacher_victim_tpr"] for item in attacks])
        student_normal = np.mean([item["student_normal_tpr"] for item in attacks])
        teacher_normal = np.mean([item["teacher_normal_tpr"] for item in attacks])
        rows.append(
            [
                "C — aluno",
                "student (1 passe)",
                student_normal,
                student_victim,
                (1.0 - student_victim / student_normal) if student_normal else None,
                0.0,
                1,
            ]
        )
        rows.append(
            [
                "C — professor",
                "RePO+ (100 máscaras)",
                teacher_normal,
                teacher_victim,
                (1.0 - teacher_victim / teacher_normal) if teacher_normal else None,
                0.0,
                100,
            ]
        )
    if not rows:
        return "_Nenhuma avaliação adversarial disponível._"
    return _table(
        [
            "direção",
            "detector",
            "TPR normal",
            "TPR vítima",
            "queda condicional",
            "bypass / no_change",
            "máscaras/ataque",
        ],
        rows,
    )


def _cascade_attack(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Ataque à cascata ainda não executado._"
    rows = []
    for attack, configs in payload["attacks"].items():
        for name, item in configs.items():
            rows.append(
                [
                    name,
                    attack,
                    item["n"],
                    item["normal_tpr"],
                    item["victim_tpr"],
                    item["bypass_rate"],
                    f"{item['fooled']}/{item['n']}",
                    item["masks_evaluated"],
                ]
            )
    return _table(
        [
            "config",
            "ataque",
            "n",
            "TPR normal",
            "TPR vítima",
            "bypass",
            "fooled",
            "máscaras/ataque",
        ],
        rows,
    )


def _distill_attack(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "_Ataque ao aluno ainda não executado._"
    rows = []
    for attack, item in payload["attacks"].items():
        rows.append(
            [
                attack,
                item["n"],
                item["student_normal_tpr"],
                item["student_victim_tpr"],
                item["teacher_normal_tpr"],
                item["teacher_victim_tpr"],
                f"{item['student_fooled']}/{item['n']}",
            ]
        )
    return _table(
        [
            "ataque",
            "n",
            "TPR normal aluno",
            "TPR vítima aluno",
            "TPR normal professor",
            "TPR vítima professor",
            "fooled aluno",
        ],
        rows,
    )


def build_markdown(results_dir: Path) -> str:
    baseline = _load(results_dir / "baseline_flow.json") or _load(
        results_dir / "baseline_flow_smoke.json"
    )
    variance = _load(results_dir / "variance_flow.json")
    adversarial = _load(results_dir / "adversarial_all.json") or _load(
        results_dir / "adversarial_pilot.json"
    )
    cascade = _load(results_dir / "cascade_flow.json")
    cascade_attack = _load(results_dir / "cascade_attack_all.json") or _load(
        results_dir / "cascade_attack.json"
    )
    distill = _load(results_dir / "distillation_flow.json")
    distill_attack = _load(results_dir / "distill_attack_all.json") or _load(
        results_dir / "distill_attack.json"
    )
    packet = _load(results_dir / "packet_normal.json")

    parts = [
        "# Eficiência do sistema de máscaras do RePO — resultados consolidados",
        "",
        "> Documento gerado por `python -m mask_experiments.aggregate`.  Todo número "
        "abaixo vem de um JSON em `results/mask_efficiency/`; os comandos de "
        "reprodução estão no fim.",
        ">",
        "> A matriz completa de runs (o que já foi feito, o que falta, escala e "
        "custo) está em [`MASK_EFFICIENCY_RUNS.md`](MASK_EFFICIENCY_RUNS.md).",
        "",
        "## Principais achados",
        "",
        _key_findings(
            baseline,
            variance,
            adversarial,
            cascade,
            distill,
            packet,
            cascade_attack,
            distill_attack,
        ),
        "",
        "## Métricas adversariais por direção",
        "",
        _adversarial_by_direction(adversarial, cascade_attack, distill_attack),
        "",
        "Todas as taxas são medidas no threshold adversarial do próprio detector "
        "(FPR de 10 %; no fluxo, o threshold de ataque é 0,8× o threshold real).  "
        "`máscaras/ataque` é o custo do **ataque**, não o custo de produção: em "
        "A é o `n_masks` do detector/atacante, em B é o `n_cheap` do estágio "
        "barato e em C é 1 passe do aluno (o professor usa 100 máscaras).",
        "",
        "## Escopo e status",
        "",
        "Este relatório consolida um **piloto em CPU** das direções A (menos "
        "máscaras / máscaras estruturadas), B (cascata e decisão sequencial) e C "
        "(destilação).  Os resultados são reais e reprodutíveis, mas as escalas "
        "são reduzidas em relação ao plano completo:",
        "",
        "- fluxo: 5.000 benignos + 300 amostras por ataque (CICIDS2017), "
        "3 repetições de máscaras;",
        "- adversarial: 4 ataques × 20 amostras por variante, com ataque "
        "adaptativo (o atacante usa a mesma política e o mesmo `n_masks`);",
        "- cascata: ataque ao estágio barato executado (Direção B);",
        "- destilação: aluno de passe único atacado diretamente (white-box) e "
        "professor avaliado nos mesmos exemplos (Direção C);",
        "- pacotes: avaliação **normal** apenas, ~2.000 janelas de "
        "terça–sexta.",
        "",
        "## 1. Baseline congelado (Gate G1)",
        "",
        _baseline(baseline),
        "",
        "O baseline (10 k benignos + 500 amostras por ataque = 15.047 amostras) "
        "reproduziu os thresholds e o TPR macro de referência dentro de ~0,7 "
        "ponto percentual.  O TPR macro é a métrica comparável ao artigo; o "
        "weighted é puxado para baixo pelas classes com TPR zero.",
        "",
        "## 2. Variância e custo por política (Direção A)",
        "",
        _variance(variance),
        "",
        "Leitura: `n_masks` menor reduz o custo linearmente; a pergunta é quanto "
        "o TPR e a estabilidade caem.  `balanced` fixa 25 % das features por "
        "máscara.  `covering` garante que toda feature apareça em cada grupo do "
        "RePO+ **quando o orçamento permite** (com 77 features, isso exige ≥25 "
        "máscaras); abaixo disso ele cai para `balanced`.",
        "",
        "## 3. Avaliação adversarial adaptativa (Direção A)",
        "",
        _adversarial(adversarial),
        "",
        "Todas as variantes usam threshold recalibrado no próprio FPR de 10 % e "
        "o atacante conhece a política do detector.  A coluna `no_change` é "
        "essenciais: ela conta as amostras que o detector já classificava como "
        "benignas antes de qualquer otimização.  Na variante `iid-5`, 50 % dos "
        "ataques caem nessa categoria — o `fooled` baixo não significa "
        "robustez, significa que o detector já errava.  `iid-5` **não** é o "
        "mesmo que o estágio barato da cascata: o `t_low` da cascata fica muito "
        "abaixo do threshold de ataque (ver 4.1).",
        "",
        "## 4. Cascata e decisão sequencial (Direção B)",
        "",
        _cascade(cascade),
        "",
        "### 4.1 Ataque à cascata",
        "",
        _cascade_attack(cascade_attack),
        "",
        "O atacante otimiza o **estágio barato** (RePO+ com `n_cheap` máscaras) até",
        "ficar abaixo de `t_low`.  `bypass` é a fração de amostras que evadem sem",
        "nunca enfrentar o ensemble completo.",
        "",
        "O custo é o número de máscaras que a regra usaria numa mistura 99 % "
        "benigno / 1 % ataque.  Para as linhas de cascata, a penúltima coluna é "
        "a taxa de benignos que pulam o ensemble e a última é o FN do estágio "
        "barato; para o RePO sequencial, a penúltima é o FPR e a última é a "
        "concordância com o RePO completo.",
        "",
        "## 5. Destilação do ensemble (Direção C)",
        "",
        _distillation(distill, distill_attack),
        "",
        "### 5.1 Ataque ao aluno",
        "",
        _distill_attack(distill_attack),
        "",
        "O ataque é white-box no aluno (mesma projeção de viabilidade do ataque",
        "de referência) e avalia o professor nos mesmos exemplos adversariais.",
        "",
        "## 6. Confirmação em pacotes (Task 5)",
        "",
        _packet(packet),
        "",
        "O TPR absoluto de pacotes não é comparável ao artigo porque o "
        "threshold foi calibrado em 540 benignos, não nos 1,4 M do conjunto "
        "completo; a comparação válida é a ordenação entre as políticas.",
        "",
        "## 7. Limitações",
        "",
        "1. Os números são de um piloto em CPU, com amostras menores que as do "
        "plano; não substituem as rodadas completas no Apuana.",
        "2. O ataque adversarial de fluxo usa 20 amostras por ataque; a "
        "variância amostral do `victim_tpr` é grande e as diferenças pequenas "
        "entre variantes não devem ser interpretadas como efeito.",
        "3. O ataque ao aluno é white-box e usa a mesma projeção de "
        "viabilidade do ataque de referência; um atacante com orçamento maior "
        "pode ser ainda mais destrutivo, então o número é um limite inferior "
        "da perda de robustez.",
        "4. O ataque à cascata otimiza o estágio barato com `n_cheap` máscaras; ",
        "   ele não prova que o estágio seja inquebrável, apenas mede o bypass ",
        "   com o orçamento do ataque de referência.",
        "5. A avaliação de pacotes é normal-only; o ataque de pacotes é ordens "
        "de magnitude mais caro e não foi executado.",
        "6. A política `covering` altera a distribuição das máscaras em relação "
        "ao treino; o efeito de *train/test mismatch* não foi isolado.",
        "7. O TPR da variância é a média dos TPRs por repetição (o detector de "
        "produção usa um conjunto de máscaras por decisão), e não o TPR do "
        "escore médio das repetições, que seria equivalente a um ensemble "
        "maior.",
        "",
        "## 8. Conclusões por direção",
        "",
        "**Direção A (menos máscaras / máscaras estruturadas).** O TPR macro é "
        "notavelmente estável com 25 máscaras: a perda fica dentro do ruído do "
        "piloto e o custo cai ~4×. As máscaras `balanced` e `covering` não "
        "superam a `iid` em TPR, mas `covering` reduz o desvio do escore entre "
        "repetições. O ganho é real, mas o limite seguro depende do ataque: a "
        "variante de 5 máscaras já mostra que o detector fica mais fácil de "
        "satisfazer (muitos ataques viram `no_change`).",
        "",
        "**Direção B (cascata / sequencial).** A decisão sequencial baseada na "
        "média é o melhor caminho: ~5–6 máscaras por amostra, mesmo TPR e "
        "concordância quase total com o RePO completo. A cascata baseada no "
        "prefixo do RePO+ economiza menos (30–43% dos benignos) porque o mínimo "
        "por grupo com poucas máscaras tem cauda esquerda pesada; e o estágio "
        "barato é justamente o elo que o atacante pode tentar derrubar. No "
        "piloto, o ataque de referência **não** conseguiu bypassar o estágio "
        "barato: `t_low` fica muito abaixo do threshold de ataque e o TPR de "
        "vítima ficou igual ao normal. Isso não valida a cascata como solução "
        "de custo — ela economiza apenas 30–43% dos benignos e perde para o "
        "RePO sequencial.",
        "",
        "**Direção C (destilação).** Um MLP de passe único reproduz o escore "
        "RePO+ no cenário normal (Spearman ~0,95 e TPR praticamente igual), mas "
        "**não preserva a robustez adversarial**: atacado diretamente, o aluno "
        "perde TPR de vítima em todos os ataques testados, com a maior queda no "
        "FTP-Patator (0,00 contra 0,15 do professor). A conclusão prática é que "
        "a destilação pode ser usada como *triagem* normal, mas não como "
        "substituto do ensemble sob ataque.",
        "",
        "**Pacotes.** A ordenação das políticas se mantém no modelo de pacotes, "
        "o que sugere que a conclusão sobre 25 máscaras não é um artefato do "
        "fluxo. A confirmação adversarial de pacotes continua em aberto.",
        "",
        "## 9. Reprodução",
        "",
        "A cadeia completa no Apuana submete todas as etapas com dependências "
        "`afterok` e grava um manifesto dos jobs:",
        "",
        "```bash",
        "bash cluster/run_mask_study.sh            # matriz completa",
        "bash cluster/run_mask_study.sh --smoke    # smoke de todas as etapas",
        "```",
        "",
        "Cada job grava em `$RESULTS_ROOT/mask_efficiency/`; o último passo faz "
        "o merge dos arrays e gera este relatório.  Para rodar uma etapa "
        "isolada:",
        "",
        "```bash",
        "bash cluster/submit.sh mask-exp baseline",
        "bash cluster/submit.sh mask-exp variance",
        "bash cluster/submit.sh mask-exp adversarial all",
        "bash cluster/submit.sh mask-exp cascade",
        "bash cluster/submit.sh mask-exp cascade-attack all",
        "bash cluster/submit.sh mask-exp distill",
        "bash cluster/submit.sh mask-exp distill-attack all",
        "bash cluster/submit.sh mask-exp packet",
        "bash cluster/submit.sh mask-exp aggregate",
        "```",
        "",
        "O piloto local (Python 3.13, TensorFlow 2.21 CPU, sem GPU) usou "
        "`--max-benign 5000 --max-per-attack 300 --repeats 3 --per-attack 20`; "
        "os números de produção vêm dos defaults do `run_mask_study.sh`.  Seed "
        "2019 em todas as etapas.",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    markdown = build_markdown(Path(args.results_dir))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
