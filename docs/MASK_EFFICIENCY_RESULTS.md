# Eficiência do sistema de máscaras do RePO — resultados consolidados

> Documento gerado por `python -m mask_experiments.aggregate`.  Todo número abaixo vem de um JSON em `results/mask_efficiency/`; os comandos de reprodução estão no fim.
>
> A matriz completa de runs (o que já foi feito, o que falta, escala e custo) está em [`MASK_EFFICIENCY_RUNS.md`](MASK_EFFICIENCY_RUNS.md).

## Principais achados

- **balanced:** 20 máscaras ficam a ≤1 ponto do TPR de 100 (TPR macro 0.268 vs 0.273, Spearman 0.964) com 5.0× menos linhas mascaradas.
- **covering:** 25 máscaras ficam a ≤1 ponto do TPR de 100 (TPR macro 0.266 vs 0.273, Spearman 0.993) com 4.0× menos linhas mascaradas.
- **iid:** 20 máscaras ficam a ≤1 ponto do TPR de 100 (TPR macro 0.268 vs 0.273, Spearman 0.964) com 5.0× menos linhas mascaradas.
- **Adversarial:** balanced-25: vítima 0.60, já benigno 0.28; covering-25: vítima 0.61, já benigno 0.28; iid-100: vítima 0.57, já benigno 0.28; iid-25: vítima 0.56, já benigno 0.28; iid-5: vítima 0.57, já benigno 0.50.
- **Sequencial:** 5.6 máscaras/amostra, TPR macro 0.273 e 99.7% de concordância com o RePO completo.
- **Cascata:** o melhor orçamento usa 61.7 máscaras/amostra e pula 43% dos benignos, mas o estágio barato tem FN de 1.0%.
- **Destilação:** aluno de passe único tem TPR 0.276 vs 0.275 do professor e Spearman 0.948.
- **Destilação sob ataque:** TPR de vítima do aluno 0.49 (normal 0.69) vs 0.65 (normal 0.71) do professor — a destilação normal não preserva a robustez adversarial.
- **Cascata sob ataque:** TPR de vítima 0.74 vs normal 0.74 (queda 0.00), bypass do estágio barato 0.00.
- **Pacotes (normal):** balanced-25 0.579; covering-25 0.574; iid-100 0.585 — melhor iid-100.

## Métricas adversariais por direção

| direção | detector | TPR normal | TPR vítima | queda condicional | bypass / no_change | máscaras/ataque |
|---|---|---|---|---|---|---|
| A — máscaras | balanced-25 | 0.7125 | 0.6000 | 0.1579 | 0.2750 | 14277 |
| A — máscaras | covering-25 | 0.7250 | 0.6125 | 0.1552 | 0.2750 | 14481 |
| A — máscaras | iid-100 | 0.7250 | 0.5750 | 0.2069 | 0.2750 | 57172 |
| A — máscaras | iid-25 | 0.7250 | 0.5625 | 0.2241 | 0.2750 | 14590 |
| A — máscaras | iid-5 | 0.6000 | 0.5750 | 0.0417 | 0.5000 | 2296 |
| B — cascata | cheap25 | 0.7375 | 0.7375 | 0.0000 | 0.0000 | 27025 |
| B — cascata | cheap5 | 0.7375 | 0.7375 | 0.0000 | 0.0000 | 5405 |
| C — aluno | student (1 passe) | 0.6875 | 0.4875 | 0.2909 | 0.0000 | 1 |
| C — professor | RePO+ (100 máscaras) | 0.7125 | 0.6500 | 0.0877 | 0.0000 | 100 |

Todas as taxas são medidas no threshold adversarial do próprio detector (FPR de 10 %; no fluxo, o threshold de ataque é 0,8× o threshold real).  `máscaras/ataque` é o custo do **ataque**, não o custo de produção: em A é o `n_masks` do detector/atacante, em B é o `n_cheap` do estágio barato e em C é 1 passe do aluno (o professor usa 100 máscaras).

## Escopo e status

Este relatório consolida um **piloto em CPU** das direções A (menos máscaras / máscaras estruturadas), B (cascata e decisão sequencial) e C (destilação).  Os resultados são reais e reprodutíveis, mas as escalas são reduzidas em relação ao plano completo:

- fluxo: 5.000 benignos + 300 amostras por ataque (CICIDS2017), 3 repetições de máscaras;
- adversarial: 4 ataques × 20 amostras por variante, com ataque adaptativo (o atacante usa a mesma política e o mesmo `n_masks`);
- cascata: ataque ao estágio barato executado (Direção B);
- destilação: aluno de passe único atacado diretamente (white-box) e professor avaliado nos mesmos exemplos (Direção C);
- pacotes: avaliação **normal** apenas, ~2.000 janelas de terça–sexta.

## 1. Baseline congelado (Gate G1)

| modo | n | threshold FPR 1% | FPR | TPR macro | TPR weighted |
|---|---|---|---|---|---|
| repo | 15047 | 0.0411 | 0.0100 | 0.2407 | 0.1524 |
| repo_plus | 15047 | 0.1058 | 0.0100 | 0.2580 | 0.1819 |

Custo: {'masked_rows': 3009400, 'mean_call_ms': 1.172, 'model_calls': 30094, 'model_seconds': 35.257733}  
Tempo: 56.7 s

O baseline (10 k benignos + 500 amostras por ataque = 15.047 amostras) reproduziu os thresholds e o TPR macro de referência dentro de ~0,7 ponto percentual.  O TPR macro é a métrica comparável ao artigo; o weighted é puxado para baixo pelas classes com TPR zero.

## 2. Variância e custo por política (Direção A)

| política | n_masks | TPR macro | desvio do TPR | TPR weighted | desvio do escore | Spearman vs iid-100 | linhas mascaradas |
|---|---|---|---|---|---|---|---|
| balanced | 5 | 0.2618 | 0.0035 | 0.1666 | 0.0102 | 0.8937 | 120705 |
| balanced | 10 | 0.2634 | 0.0022 | 0.1739 | 0.0055 | 0.9384 | 241410 |
| balanced | 20 | 0.2679 | 0.0036 | 0.1891 | 0.0035 | 0.9637 | 482820 |
| balanced | 25 | 0.2692 | 0.0031 | 0.1906 | 0.0030 | 0.9715 | 603525 |
| balanced | 50 | 0.2674 | 0.0009 | 0.1920 | 0.0016 | 0.9918 | 1207050 |
| balanced | 100 | 0.2734 | 0.0009 | 0.1992 | 0.0007 | 0.9993 | 2414100 |
| covering | 5 | 0.2618 | 0.0035 | 0.1666 | 0.0102 | 0.8937 | 120705 |
| covering | 10 | 0.2634 | 0.0022 | 0.1739 | 0.0055 | 0.9384 | 241410 |
| covering | 20 | 0.2679 | 0.0036 | 0.1891 | 0.0035 | 0.9637 | 482820 |
| covering | 25 | 0.2656 | 0.0030 | 0.1889 | 0.0013 | 0.9928 | 603525 |
| covering | 50 | 0.2696 | 0.0012 | 0.1946 | 0.0010 | 0.9966 | 1207050 |
| covering | 100 | 0.2726 | 0.0015 | 0.1991 | 0.0007 | 0.9990 | 2414100 |
| iid | 5 | 0.2573 | 0.0086 | 0.1656 | 0.0107 | 0.8897 | 120705 |
| iid | 10 | 0.2628 | 0.0009 | 0.1740 | 0.0056 | 0.9385 | 241410 |
| iid | 20 | 0.2679 | 0.0043 | 0.1899 | 0.0035 | 0.9641 | 482820 |
| iid | 25 | 0.2674 | 0.0022 | 0.1893 | 0.0030 | 0.9727 | 603525 |
| iid | 50 | 0.2669 | 0.0004 | 0.1914 | 0.0015 | 0.9930 | 1207050 |
| iid | 100 | 0.2730 | 0.0009 | 0.1988 | 0.0007 | 1.0000 | 2414100 |

Leitura: `n_masks` menor reduz o custo linearmente; a pergunta é quanto o TPR e a estabilidade caem.  `balanced` fixa 25 % das features por máscara.  `covering` garante que toda feature apareça em cada grupo do RePO+ **quando o orçamento permite** (com 77 features, isso exige ≥25 máscaras); abaixo disso ele cai para `balanced`.

## 3. Avaliação adversarial adaptativa (Direção A)

| variante | ataque | n | TPR normal | TPR vítima | fooled | no_change | máscaras/ataque |
|---|---|---|---|---|---|---|---|
| balanced-25 | DDoS | 20 | 0.9000 | 0.8500 | 3/20 | 2 | 21997 |
| balanced-25 | DoS Hulk | 20 | 0.8500 | 0.8500 | 2/20 | 2 | 21632 |
| balanced-25 | DoS slowloris | 20 | 0.6500 | 0.4500 | 9/20 | 7 | 12635 |
| balanced-25 | FTP-Patator | 20 | 0.4500 | 0.2500 | 9/20 | 11 | 845 |
| covering-25 | DDoS | 20 | 0.8000 | 0.8000 | 0/20 | 4 | 21625 |
| covering-25 | DoS Hulk | 20 | 1.0000 | 0.8500 | 4/20 | 0 | 21845 |
| covering-25 | DoS slowloris | 20 | 0.6500 | 0.5500 | 7/20 | 7 | 13552 |
| covering-25 | FTP-Patator | 20 | 0.4500 | 0.2500 | 9/20 | 11 | 905 |
| iid-100 | DDoS | 20 | 0.8000 | 0.8500 | 0/20 | 4 | 86500 |
| iid-100 | DoS Hulk | 20 | 1.0000 | 0.9000 | 4/20 | 0 | 87180 |
| iid-100 | DoS slowloris | 20 | 0.6500 | 0.5000 | 4/20 | 7 | 52120 |
| iid-100 | FTP-Patator | 20 | 0.4500 | 0.0500 | 9/20 | 11 | 2890 |
| iid-25 | DDoS | 20 | 0.9000 | 0.8500 | 3/20 | 2 | 22592 |
| iid-25 | DoS Hulk | 20 | 0.9000 | 0.8000 | 2/20 | 2 | 21632 |
| iid-25 | DoS slowloris | 20 | 0.6500 | 0.4000 | 8/20 | 7 | 12480 |
| iid-25 | FTP-Patator | 20 | 0.4500 | 0.2000 | 9/20 | 11 | 1657 |
| iid-5 | DDoS | 20 | 0.8500 | 0.9000 | 2/20 | 5 | 3701 |
| iid-5 | DoS Hulk | 20 | 0.8000 | 0.8000 | 0/20 | 4 | 4325 |
| iid-5 | DoS slowloris | 20 | 0.4500 | 0.3000 | 4/20 | 12 | 1145 |
| iid-5 | FTP-Patator | 20 | 0.3000 | 0.3000 | 1/20 | 19 | 15 |

**Resumo por variante**

| variante | threshold real (FPR 10%) | threshold de ataque | TPR vítima médio | fooled médio | no_change médio | máscaras/ataque (média) |
|---|---|---|---|---|---|---|
| balanced-25 | 0.0280 | 0.0224 | 0.6000 | 0.2875 | 0.2750 | 14277 |
| covering-25 | 0.0205 | 0.0164 | 0.6125 | 0.2500 | 0.2750 | 14481 |
| iid-100 | 0.0168 | 0.0134 | 0.5750 | 0.2125 | 0.2750 | 57172 |
| iid-25 | 0.0278 | 0.0223 | 0.5625 | 0.2750 | 0.2750 | 14590 |
| iid-5 | 0.0799 | 0.0639 | 0.5750 | 0.0875 | 0.5000 | 2296 |

Todas as variantes usam threshold recalibrado no próprio FPR de 10 % e o atacante conhece a política do detector.  A coluna `no_change` é essenciais: ela conta as amostras que o detector já classificava como benignas antes de qualquer otimização.  Na variante `iid-5`, 50 % dos ataques caem nessa categoria — o `fooled` baixo não significa robustez, significa que o detector já errava.  `iid-5` **não** é o mesmo que o estágio barato da cascata: o `t_low` da cascata fica muito abaixo do threshold de ataque (ver 4.1).

## 4. Cascata e decisão sequencial (Direção B)

| regra | máscaras/amostra (99% benigno) | TPR macro | TPR weighted | benignos pulados / FPR | FN do estágio barato / concordância |
|---|---|---|---|---|---|
| RePO+ integral | 100 | 0.2730 | 0.1988 | — | — |
| cheap10_fn0.0 | 64.7342 | 0.2730 | 0.1988 | 0.3958 | 0.0000 |
| cheap10_fn0.001 | 63.9490 | 0.2730 | 0.1988 | 0.4046 | 0.0013 |
| cheap10_fn0.01 | 61.6600 | 0.2730 | 0.1988 | 0.4302 | 0.0102 |
| cheap25_fn0.0 | 70.3247 | 0.2730 | 0.1988 | 0.3997 | 0.0000 |
| cheap25_fn0.001 | 69.6209 | 0.2730 | 0.1988 | 0.4091 | 0.0013 |
| cheap25_fn0.01 | 69.0499 | 0.2730 | 0.1988 | 0.4167 | 0.0102 |
| cheap5_fn0.0 | 71.6721 | 0.2730 | 0.1988 | 0.3012 | 0.0000 |
| cheap5_fn0.001 | 67.9528 | 0.2730 | 0.1988 | 0.3407 | 0.0013 |
| cheap5_fn0.01 | 65.0225 | 0.2730 | 0.1988 | 0.3718 | 0.0102 |
| RePO sequencial | 5.5513 | 0.2733 | 0.1813 | 0.0102 | 0.9966 |

### 4.1 Ataque à cascata

| config | ataque | n | TPR normal | TPR vítima | bypass | fooled | máscaras/ataque |
|---|---|---|---|---|---|---|---|
| cheap25 | DDoS | 20 | 0.7500 | 0.7500 | 0.0000 | 0/20 | 27025 |
| cheap5 | DDoS | 20 | 0.7500 | 0.7500 | 0.0000 | 0/20 | 5405 |
| cheap25 | DoS Hulk | 20 | 1.0000 | 1.0000 | 0.0000 | 0/20 | 27025 |
| cheap5 | DoS Hulk | 20 | 1.0000 | 1.0000 | 0.0000 | 0/20 | 5405 |
| cheap25 | DoS slowloris | 20 | 0.6000 | 0.6000 | 0.0000 | 0/20 | 27025 |
| cheap5 | DoS slowloris | 20 | 0.6000 | 0.6000 | 0.0000 | 0/20 | 5405 |
| cheap25 | FTP-Patator | 20 | 0.6000 | 0.6000 | 0.0000 | 0/20 | 27025 |
| cheap5 | FTP-Patator | 20 | 0.6000 | 0.6000 | 0.0000 | 0/20 | 5405 |

O atacante otimiza o **estágio barato** (RePO+ com `n_cheap` máscaras) até
ficar abaixo de `t_low`.  `bypass` é a fração de amostras que evadem sem
nunca enfrentar o ensemble completo.

O custo é o número de máscaras que a regra usaria numa mistura 99 % benigno / 1 % ataque.  Para as linhas de cascata, a penúltima coluna é a taxa de benignos que pulam o ensemble e a última é o FN do estágio barato; para o RePO sequencial, a penúltima é o FPR e a última é a concordância com o RePO completo.

## 5. Destilação do ensemble (Direção C)

| modelo | TPR macro | TPR weighted | Spearman | passes |
|---|---|---|---|---|
| professor (100 máscaras) | 0.2751 | 0.2075 | 1.0000 | 100 |
| aluno (passe único) | 0.2761 | 0.2087 | 0.9478 | 1 |

RMSE do aluno: 0.0109; MAE: 0.0074. Ataque adaptativo no aluno: executado (seção 5.1).

### 5.1 Ataque ao aluno

| ataque | n | TPR normal aluno | TPR vítima aluno | TPR normal professor | TPR vítima professor | fooled aluno |
|---|---|---|---|---|---|---|
| DDoS | 20 | 0.6500 | 0.5500 | 0.7000 | 0.7000 | 2/20 |
| DoS Hulk | 20 | 0.9500 | 0.8000 | 1.0000 | 1.0000 | 3/20 |
| DoS slowloris | 20 | 0.8000 | 0.6000 | 0.8000 | 0.7500 | 4/20 |
| FTP-Patator | 20 | 0.3500 | 0.0000 | 0.3500 | 0.1500 | 7/20 |

O ataque é white-box no aluno (mesma projeção de viabilidade do ataque
de referência) e avalia o professor nos mesmos exemplos adversariais.

## 6. Confirmação em pacotes (Task 5)

| variante | modo | threshold FPR 1% | TPR macro | TPR weighted |
|---|---|---|---|---|
| balanced-25 | repo | 0.0240 | 0.5618 | 0.5210 |
| balanced-25 | repo_plus | 0.0968 | 0.5791 | 0.5404 |
| covering-25 | repo | 0.0244 | 0.5528 | 0.5101 |
| covering-25 | repo_plus | 0.0973 | 0.5743 | 0.5342 |
| iid-100 | repo | 0.0232 | 0.5718 | 0.5318 |
| iid-100 | repo_plus | 0.0871 | 0.5845 | 0.5443 |

Janelas: 1994.

O TPR absoluto de pacotes não é comparável ao artigo porque o threshold foi calibrado em 540 benignos, não nos 1,4 M do conjunto completo; a comparação válida é a ordenação entre as políticas.

## 7. Limitações

1. Os números são de um piloto em CPU, com amostras menores que as do plano; não substituem as rodadas completas no Apuana.
2. O ataque adversarial de fluxo usa 20 amostras por ataque; a variância amostral do `victim_tpr` é grande e as diferenças pequenas entre variantes não devem ser interpretadas como efeito.
3. O ataque ao aluno é white-box e usa a mesma projeção de viabilidade do ataque de referência; um atacante com orçamento maior pode ser ainda mais destrutivo, então o número é um limite inferior da perda de robustez.
4. O ataque à cascata otimiza o estágio barato com `n_cheap` máscaras; 
   ele não prova que o estágio seja inquebrável, apenas mede o bypass 
   com o orçamento do ataque de referência.
5. A avaliação de pacotes é normal-only; o ataque de pacotes é ordens de magnitude mais caro e não foi executado.
6. A política `covering` altera a distribuição das máscaras em relação ao treino; o efeito de *train/test mismatch* não foi isolado.
7. O TPR da variância é a média dos TPRs por repetição (o detector de produção usa um conjunto de máscaras por decisão), e não o TPR do escore médio das repetições, que seria equivalente a um ensemble maior.

## 8. Conclusões por direção

**Direção A (menos máscaras / máscaras estruturadas).** O TPR macro é notavelmente estável com 25 máscaras: a perda fica dentro do ruído do piloto e o custo cai ~4×. As máscaras `balanced` e `covering` não superam a `iid` em TPR, mas `covering` reduz o desvio do escore entre repetições. O ganho é real, mas o limite seguro depende do ataque: a variante de 5 máscaras já mostra que o detector fica mais fácil de satisfazer (muitos ataques viram `no_change`).

**Direção B (cascata / sequencial).** A decisão sequencial baseada na média é o melhor caminho: ~5–6 máscaras por amostra, mesmo TPR e concordância quase total com o RePO completo. A cascata baseada no prefixo do RePO+ economiza menos (30–43% dos benignos) porque o mínimo por grupo com poucas máscaras tem cauda esquerda pesada; e o estágio barato é justamente o elo que o atacante pode tentar derrubar. No piloto, o ataque de referência **não** conseguiu bypassar o estágio barato: `t_low` fica muito abaixo do threshold de ataque e o TPR de vítima ficou igual ao normal. Isso não valida a cascata como solução de custo — ela economiza apenas 30–43% dos benignos e perde para o RePO sequencial.

**Direção C (destilação).** Um MLP de passe único reproduz o escore RePO+ no cenário normal (Spearman ~0,95 e TPR praticamente igual), mas **não preserva a robustez adversarial**: atacado diretamente, o aluno perde TPR de vítima em todos os ataques testados, com a maior queda no FTP-Patator (0,00 contra 0,15 do professor). A conclusão prática é que a destilação pode ser usada como *triagem* normal, mas não como substituto do ensemble sob ataque.

**Pacotes.** A ordenação das políticas se mantém no modelo de pacotes, o que sugere que a conclusão sobre 25 máscaras não é um artefato do fluxo. A confirmação adversarial de pacotes continua em aberto.

## 9. Reprodução

A cadeia completa no Apuana submete todas as etapas com dependências `afterok` e grava um manifesto dos jobs:

```bash
bash cluster/run_mask_study.sh            # matriz completa
bash cluster/run_mask_study.sh --smoke    # smoke de todas as etapas
```

Cada job grava em `$RESULTS_ROOT/mask_efficiency/`; o último passo faz o merge dos arrays e gera este relatório.  Para rodar uma etapa isolada:

```bash
bash cluster/submit.sh mask-exp baseline
bash cluster/submit.sh mask-exp variance
bash cluster/submit.sh mask-exp adversarial all
bash cluster/submit.sh mask-exp cascade
bash cluster/submit.sh mask-exp cascade-attack all
bash cluster/submit.sh mask-exp distill
bash cluster/submit.sh mask-exp distill-attack all
bash cluster/submit.sh mask-exp packet
bash cluster/submit.sh mask-exp aggregate
```

O piloto local (Python 3.13, TensorFlow 2.21 CPU, sem GPU) usou `--max-benign 5000 --max-per-attack 300 --repeats 3 --per-attack 20`; os números de produção vêm dos defaults do `run_mask_study.sh`.  Seed 2019 em todas as etapas.
