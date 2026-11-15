# Eficiência do sistema de máscaras do RePO — consolidação interina

> Gerado em 15/11/2026 a partir dos JSONs em `$HOME/replicating-repo-results/mask_efficiency/`.
> Este documento separa explicitamente o que já está em **escala cheia** do que ainda é
> **piloto** (2 000 benignos + 200/ataque, 4 ataques).  Números de escalas diferentes não
> devem ser comparados entre si.

## 1. O que já tem escala cheia

| experimento | amostra | repetições | status |
|---|---|---|---|
| baseline RePO / RePO+ | 144 182 (100 k benignos + 5 k/ataque) | 5 | completo |
| variância normal (A) | 144 182 | 5 | completo, 3 políticas |
| adversarial (A) | 5 000 por ataque | — | **2 de 11 ataques** (FTP-Patator, SSH-Patator) |
| cascata (B) | 4 047 | 1 | piloto |
| destilação (C) | 4 047 | 1 | piloto |
| ataques a B e C | 20 por ataque | — | piloto |
| pacotes | 294 janelas | — | piloto |

Baseline em escala cheia, FPR de 1 %:

| modo | TPR macro | TPR ponderado |
|---|---|---|
| RePO | 0.2396 | 0.1581 |
| RePO+ | **0.2546** | **0.1873** |

## 2. Direção A — menos máscaras e máscaras estruturadas

### 2.1 Curva normal (escala cheia, 144 182 amostras, 5 repetições)

RePO+ com threshold recalibrado em FPR de 1 %:

| máscaras | iid macro | balanced macro | covering macro | iid pond. | balanced pond. | covering pond. |
|---|---|---|---|---|---|---|
| 5 | 0.2439 | 0.2454 | 0.2454 | 0.1565 | 0.1566 | 0.1566 |
| 10 | 0.2474 | 0.2476 | 0.2476 | 0.1649 | 0.1639 | 0.1639 |
| 20 | 0.2520 | 0.2518 | 0.2518 | 0.1774 | 0.1771 | 0.1771 |
| 25 | 0.2509 | 0.2533 | 0.2550 | 0.1792 | 0.1797 | 0.1839 |
| 50 | 0.2554 | 0.2556 | 0.2566 | 0.1857 | 0.1860 | 0.1867 |
| 100 | 0.2560 | 0.2559 | 0.2560 | 0.1871 | 0.1871 | 0.1871 |

Leitura:

- **O joelho está em 25 máscaras.** `covering-25` fica em 0.2550 de macro (−0.4 % relativo)
  e 0.1839 de ponderado (−1.7 %) contra 100 máscaras, com **25 % do custo de mascaramento**.
- Abaixo disso o custo cresce: em 10 máscaras o ponderado cai 12 %; em 5, 16 %.
- **A política quase não importa nesta escala.** `iid`, `balanced` e `covering` empatam
  dentro do ruído (desvio típico ≈ 0.001–0.005). `covering` só se diferencia a partir de
  25 máscaras, porque abaixo disso o grupo não tem slots suficientes para cobrir as 77
  features e a política degrada para `balanced` (comportamento documentado no código).
- O TPR **ponderado** é bem mais sensível que o macro: as classes grandes (DDoS, DoS Hulk)
  perdem mais com poucas máscaras.

### 2.2 Ataque adaptativo (escala cheia, FTP-Patator, 5 000 amostras)

Métricas no threshold de ataque (0.8× o threshold real), como definido no protocolo:

| variante | TPR normal | TPR vítima | queda relativa | fooled |
|---|---|---|---|---|
| iid-100 | 0.4982 | 0.1764 | **−64.6 %** | 2 490/5 000 |
| iid-50 | 0.4982 | 0.2044 | −59.0 % | 2 492/5 000 |
| balanced-50 | 0.4982 | 0.2054 | −58.8 % | 2 491/5 000 |
| iid-25 | 0.4966 | 0.2454 | −50.6 % | 2 438/5 000 |
| covering-25 | 0.4984 | 0.2474 | −50.4 % | 2 478/5 000 |
| balanced-25 | 0.4966 | 0.2552 | −48.6 % | 2 436/5 000 |
| iid-10 | 0.4006 | 0.3564 | **−11.0 %** | 1 873/5 000 |
| balanced-10 | 0.4070 | 0.3570 | −12.3 % | 1 880/5 000 |
| iid-5 | 0.3318 | 0.3442 | +3.7 % (sem queda) | 1 363/5 000 |

Leitura:

- **Mais máscaras = mais superfície de ataque.** O atacante tem `n_masks` sorteios para
  achar uma configuração que engane o ensemble; com 100 máscaras ele derruba o TPR de
  0.50 para 0.18. Com 10, a queda é de 11 %; com 5, não há queda mensurável.
- O joelho de **custo** (25) não é o joelho de **robustez** (10). `iid-10` entrega
  89 % de retenção sob ataque contra 35 % do `iid-100`, mas paga ~10 pontos de TPR normal
  (0.40 vs 0.50) no threshold de ataque.
- `covering-25` retém 50 % contra 35 % do `iid-100`, mantendo o melhor TPR normal da faixa
  de 25 — é o melhor compromisso custo × acurácia × robustez **entre as variantes de 25**.

**Ressalva importante:** FTP-Patator tem TPR **0.000 no threshold de produção (1 % FPR)**
— o detector simplesmente não detecta essa classe nesse ponto de operação.  As comparações
acima valem para o regime de threshold de ataque (FPR ≈ 10 %), que é o protocolo do estudo,
não para a operação a 1 %.

**SSH-Patator não é informativo:** o TPR normal no threshold de ataque varia de 0.005 a
0.059, então as quedas observadas (algumas até negativas) são ruído de amostragem.

## 3. Direção B — cascata e decisão sequencial (piloto)

| configuração | máscaras/amostra | pula benignos | TPR macro | FN do estágio barato |
|---|---|---|---|---|
| RePO+ completo (referência) | 100.0 | — | 0.2379 | — |
| cascata cheap=25, FN alvo 0 % | 70.1 | 40.3 % | 0.2379 | 0.0 % |
| cascata cheap=25, FN alvo 1 % | 69.0 | 41.8 % | 0.2379 | 1.03 % |
| cascata cheap=5, FN alvo 0 % | 68.5 | 33.5 % | 0.2379 | 0.0 % |
| cascata cheap=5, FN alvo 1 % | **64.9** | 37.3 % | 0.2379 | 1.03 % |
| sequencial RePO | **5.3** | — | 0.2170 | — |

Leitura provisória:

- A cascata economiza **30–35 % das máscaras** sem perda de TPR macro **na amostra do
  piloto**, mas o orçamento de 1 % de FN é consumido inteiro pelo estágio barato.
- A decisão sequencial é muito mais agressiva: **5.3 máscaras/amostra** (≈ 95 % menos),
  99.93 % de concordância com o RePO completo, ao custo de −2.1 pontos de macro.
- Sob o ataque do piloto (20 amostras/ataque), a cascata **não foi burlada**
  (bypass 0.00; TPR de vítima 0.71 = normal).  Isso é encorajador, mas a amostra é pequena
  demais para concluir robustez.

## 4. Direção C — destilação (piloto)

| detector | TPR normal | TPR vítima | Spearman vs professor | passes |
|---|---|---|---|---|
| professor RePO+ (100 máscaras) | 0.6875 | 0.6750 | — | 100 |
| aluno destilado | 0.7000 | **0.3625** | 0.921 | **1** |

Por ataque (20 exemplos cada), o padrão se repete:

| ataque | aluno normal → vítima | professor normal → vítima |
|---|---|---|
| DDoS | 0.600 → 0.300 | 0.550 → 0.600 |
| DoS Hulk | 1.000 → 0.850 | 1.000 → 1.000 |
| DoS slowloris | 0.750 → 0.250 | 0.750 → 0.750 |
| FTP-Patator | 0.450 → 0.050 | 0.450 → 0.350 |

Leitura provisória, mas já bastante clara:

- **Normal:** o aluno de passe único iguala/ultrapassa o professor (0.263 vs 0.241 de macro
  na avaliação normal, Spearman 0.921) com **1 % do custo** de máscaras.
- **Sob ataque:** o aluno **perde 48 % do TPR** enquanto o professor perde 1.8 %.
  A destilação comprime a função de escore, mas também comprime a diversidade de máscaras
  que dava robustez ao professor — o aluno é white-box e não tem ensemble para se apoiar.
- É o resultado direcional mais forte até agora: **destilação é ganho de custo, não de
  robustez**.

## 5. Pacotes (piloto)

294 janelas, apenas tráfego normal, RePO+ a ~1 % FPR:

| variante | TPR macro |
|---|---|
| balanced-25 | **0.634** |
| iid-100 | 0.615 |
| covering-25 | 0.603 |

A ordenação é a mesma do fluxo: 25 máscaras estruturadas não perdem para 100 iid.

## 6. Conclusões que já sustentam

1. **25 máscaras estruturadas substituem 100 iid** com perda ≤ 0.5 ponto de TPR macro e
   ≤ 2 % no ponderado, em 144 k amostras e 5 repetições.  A política (`iid`/`balanced`/
   `covering`) é irrelevante no topo; `covering-25` é a melhor da faixa.
2. **Reduzir máscaras aumenta a robustez adversarial** no cenário medido: `iid-100` perde
   65 % do TPR sob FTP-Patator; `iid-10` perde 11 %; `iid-5` não perde.  Existe um trade-off
   explícito entre TPR normal e retenção sob ataque.
3. **Cascata** troca ~30 % das máscaras por um orçamento de 1 % de FN no estágio barato,
   sem perda de TPR na amostra piloto.
4. **Sequencial** chega a 5 máscaras/amostra com 99.9 % de concordância e −2.1 pontos de TPR.
5. **Destilação é ótima no normal e frágil no adversarial**: aluno de 1 passe empata com o
   professor em tráfego limpo e perde ~48 % do TPR sob ataque, contra 2 % do professor.

## 7. O que ainda pode mudar

- Faltam **9 dos 11 ataques** em escala cheia (Hulk, GoldenEye, Heartbleed, Infiltration,
  Bot, PortScan, DDoS, slowloris, Slowhttptest).  As conclusões 2 e 3 podem mudar de sinal
  quando os ataques pesados entrarem.
- B e C estão em 4 047 amostras e 4 ataques × 20 exemplos; os números de ataque a B e C
  são indicativos, não conclusivos.
- No fluxo, o TPR de produção é baixo para várias classes (FTP, SSH, PortScan e Web Attack
  ficam em 0.000 a 1 % FPR).  As conclusões adversariais valem para o threshold de ataque
  (FPR ≈ 10 %), não para a operação a 1 %.
- O relatório final (`MASK_EFFICIENCY_RESULTS.md`) é regenerado automaticamente pelo job
  `aggregate` quando os 11 ataques terminarem; este documento é a leitura interina.
