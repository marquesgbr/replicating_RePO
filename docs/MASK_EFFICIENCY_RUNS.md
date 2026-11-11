# Matriz de runs — análise completa das direções A, B e C

> Este documento consolida **todas as execuções necessárias** para a análise
> completa do sistema de máscaras do RePO e o **efeito de cada run nas
> métricas** do relatório.  Ele complementa `MASK_EFFICIENCY_RESULTS.md`
> (resultados) e o plano em `docs/superpowers/plans/2026-11-10-mask-efficiency-analysis.md`
> (desenho).  Nada aqui foi executado no Apuana ainda: a coluna **status** diz
> o que já existe no piloto local e o que precisa rodar em produção.

## 0. O que conta como "análise completa"

Uma direção só está fechada quando tem, **na mesma amostra e no mesmo FPR**:

1. **Custo de produção** medido (linhas mascaradas, chamadas ao modelo, tempo),
   não estimado.
2. **TPR normal** no FPR de 1 %, com threshold recalibrado.
3. **TPR de vítima** no FPR de 10 %, com ataque adaptativo que conhece a
   política e o número de máscaras do detector.
4. **Taxa de `fooled` condicional** (entre as amostras que o detector acertava
   antes do ataque) e a taxa de `no_change` (já benignas) reportadas
   separadamente.
5. **Custo do ataque** (máscaras/forward passes por amostra), para não comparar
   um ataque caro contra um detector barato.

O piloto local cumpriu o formato, mas com **20 amostras por ataque**: a margem
de erro binomial é ≈ 11 pontos percentuais em TPR ≈ 0,5.  As diferenças de
1–5 pp do piloto na Direção A **não são conclusivas**.  A rodada de produção
(5 k por ataque) reduz a margem para ≈ 0,7 pp e é o que decide a comparação.

## 1. Constantes do protocolo (não variam entre runs)

| Item | Valor | Onde |
|---|---|---|
| Dataset principal | CICIDS2017 fluxo, dias Tue–Fri | `code/mask_experiments/data.py` |
| Dataset de confirmação | CICIDS2017 pacotes, 2 000 janelas balanceadas | `code/mask_experiments/packet.py` |
| Seed | 2019 | `SEED` |
| Amostragem | aleatória por classe, não as primeiras N | `sample_flow_test` |
| FPR normal | 1 % | `threshold_at_fpr(..., 0.01)` |
| FPR adversarial | 10 % | `PHASE_VICTIM` |
| Threshold de ataque (fluxo) | 0,8 × threshold real | `ATTACK_FACTOR` |
| Ataque | gradiente, projeção de viabilidade, política e `n_masks` do detector | `new_data_eval/attack.py` |
| Custo | linhas mascaradas, chamadas, segundos | `ScoringCost` |
| Escala de produção | 100 k benignos + 5 k/ataque (11 ataques) | `MASK_MAX_*` |

`Heartbleed` tem 11 amostras e `Infiltration` 36 no teste; as duas entram no
TPR macro com o peso de uma classe, então o intervalo de confiança delas é
grande e deve aparecer no relatório (não esconder na média).

## 2. Tabela mestra

Legenda: ✅ feito no piloto · 🔁 refazer em produção · 🧩 precisa de código
novo · ⛔ fora de escopo por custo, com justificativa.

| ID | Direção | Comando | Escala de produção | Saída | Status |
|---|---|---|---|---|---|
| R0 | controle | `mask-exp baseline` | 100 k + 5 k/ataque | `baseline_flow.json` | 🔁 (piloto ✅) |
| R1 | A — normal | `mask-exp variance` | 6 `n_masks` × 3 políticas × 5 repetições | `variance_flow.json` | 🔁 (piloto ✅, 3 rep.) |
| R2 | A — adversarial | `mask-exp adversarial all` | 9 variantes × 11 ataques × 5 k | `adversarial/<ataque>.json` | 🔁 (piloto ✅, 4 var. × 4 ataques × 20) |
| R3 | B — normal | `mask-exp cascade` | `cheap ∈ {5,10,25}` × `fn ∈ {0;0,1%;1%}` × 5 rep. | `cascade_flow.json` | 🔁 (piloto ✅, 3 rep.) |
| R4 | B — adversarial | `mask-exp cascade-attack all` | `cheap ∈ {5,10,25}` × 11 ataques × 5 k | `cascade_attack/<ataque>.json` | 🔁 (piloto ✅, 2 cheap × 4 ataques × 20) |
| R5 | C — treino | `mask-exp distill` | 100 k + 5 k/ataque, 40 épocas | `distillation_flow.json` + `.keras` | 🔁 (piloto ✅) |
| R5b | C — controle | novo `--student masked_ae` | idem R5 | `distillation_masked_ae.json` | 🧩 |
| R6 | C — adversarial | `mask-exp distill-attack all` | 11 ataques × 5 k, professor pareado | `distill_attack/<ataque>.json` | 🔁 (piloto ✅, 4 × 20) |
| R7 | pacotes — normal | `mask-exp packet` | 2 000 janelas × 3 variantes | `packet_normal.json` | 🔁 (piloto ✅) |
| R8 | pacotes — adversarial | novo `mask-exp packet-attack` | 11 ataques × N janelas | `packet_attack/<ataque>.json` | 🧩⛔ |
| R9 | retreino condicional | `train_flow_based_model.py --mask-policy ...` | só se G2 falhar | modelo + avaliação | 🧩 condicional |
| R10 | agregação | `mask-exp aggregate` | merge + Markdown | `MASK_EFFICIENCY_RESULTS.md` | ✅ |

## 3. Detalhe de cada run

### R0 — Baseline congelado (controle)

```bash
bash cluster/submit.sh mask-exp baseline
```

**Para que serve:** congela thresholds e TPR do protocolo original na mesma
amostra usada pelas variantes.  Sem ele, qualquer diferença de 1 pp pode ser
atribuída à amostragem.

**Métricas que fixa:** threshold RePO/RePO+ no FPR 1 %, TPR macro e weighted,
custo de 100 máscaras.  **Gate G1:** se não reproduzir `REPLICATION_ANALYSIS.md`
dentro do ruído, parar.

### R1 — Variância normal da Direção A

```bash
MASK_VARIANCE_N="5,10,20,25,50,100" \
MASK_POLICIES="iid,balanced,covering" \
MASK_REPEATS=5 \
bash cluster/submit.sh mask-exp variance
```

**Células:** 6 `n_masks` × 3 políticas × 5 repetições = 90 passagens sobre a
amostra.  As repetições são obrigatórias porque o escore RePO é estocástico;
com 3 o desvio ainda é mal estimado.

**Métricas afetadas:**

- TPR macro/weighted no FPR 1 % (curva TPR × `n_masks`).
- `score_std_across_repeats` e `score_std_p95` (estabilidade da decisão).
- Spearman contra o escore de 100 máscaras (fidelidade da variante barata).
- Custo: linhas mascaradas, chamadas e segundos por variante.
- **Produto:** a fronteira custo × TPR normal e o "joelho" em 20–25 máscaras.

**Gate G2:** se `balanced`/`covering` caírem > 1 pp no TPR normal, rodar R9
(retreino com a política) antes de descartar.

### R2 — Adversarial da Direção A (o run central)

```bash
MASK_VARIANTS="iid:100,iid:50,iid:25,iid:10,iid:5,balanced:50,balanced:25,balanced:10,covering:25" \
bash cluster/submit.sh mask-exp adversarial all
```

**Células:** 9 variantes × 11 ataques × 5 k = ~495 k execuções de ataque (menos
`Heartbleed`/`Infiltration`, limitados pelos dados).  O array roda um job por
ataque; o `aggregate` faz o merge.

**Por que essas 9:** `iid` cobre a curva em 5/10/25/50/100; `balanced` testa se
preservar exatamente 25 % das features compra robustez em 10/25/50; `covering`
só existe a partir de 25 (com menos de 25 máscaras não há cobertura possível
das 77 features).

**Métricas afetadas:**

- TPR de vítima no FPR 10 % por variante e ataque.
- Queda condicional `1 − vítima/normal` (entre as detectadas antes).
- `fooled` (evadidas + backup) e `kinds` (`evaded`, `backup`, `no_change`,
  `not_fooled`).
- **`no_change`** é a métrica de alerta: em `iid-5` do piloto, 50 % dos ataques
  já eram benignos antes do ataque; o `victim_tpr` parece bom só porque o
  detector já errava.
- Custo do ataque: `masks_evaluated` e `model_calls` por amostra.
- **Produto:** a fronteira custo × TPR de vítima (Pareto de robustez).

**Alternativa mais barata:** rodar a curva `iid` (5 variantes) em 4 ataques
representativos e só os finalistas (`iid-100`, `iid-25`, `balanced-25`,
`covering-25`) nos 11.  Isso exige um subdiretório de saída por estágio (ver
§6) para não sobrescrever o JSON do mesmo ataque; hoje o `merge` lê um único
diretório.

### R3 — Cascata e decisão sequencial (Direção B, normal)

```bash
MASK_CHEAP="5,10,25" \
MASK_FN_RATES="0.0,0.001,0.01" \
MASK_REPEATS=5 \
bash cluster/submit.sh mask-exp cascade
```

`n_cheap` precisa ser múltiplo de 5 porque o RePO+ agrupa em 5 grupos; o
`k ∈ {1,4}` do plano não se aplica ao prefixo do RePO+ (só ao RePO por média,
que não está implementado como cascata).  Isso deve ser dito no relatório.

**Métricas afetadas:**

- Máscaras/amostra na mistura 99 % benigno / 1 % ataque.
- Taxa de benignos que pulam o ensemble (`benign_skip_rate`).
- FN do estágio barato (`cheap_false_negative_rate`) e `t_low`.
- TPR macro/weighted da cascata e do RePO+ integral.
- RePO sequencial: máscaras/amostra, FPR, TPR e **concordância com o RePO
  completo**.
- **Produto:** o custo real de produção da cascata e a sensibilidade ao alvo
  de FN.

### R4 — Ataque ao estágio barato (Direção B, adversarial)

```bash
MASK_CHEAP_ATTACK="5,10,25" \
MASK_CASCADE_FN=0.01 \
MASK_CASCADE_FPR=0.10 \
bash cluster/submit.sh mask-exp cascade-attack all
```

**Células:** 3 estágios × 11 ataques × 5 k.  O atacante otimiza o escore do
estágio barato até cair abaixo de `t_low`; se conseguir, a amostra é benigna
sem nunca passar pelo ensemble completo.

**Métricas afetadas:**

- **`bypass_rate`** — a métrica de segurança da cascata.
- TPR de vítima da cascata no FPR 10 % e `fooled`.
- TPR do ensemble completo nos **mesmos** exemplos adversariais (comparação
  pareada; já é calculado).
- `masks_evaluated` do ataque (custo de derrubar o estágio barato).
- **Gate G3:** bypass alto → a cascata vira resultado negativo documentado,
  não recomendação.

### R5 — Destilação (Direção C, normal)

```bash
MASK_EPOCHS=40 bash cluster/submit.sh mask-exp distill
```

**Métricas afetadas:**

- TPR macro/weighted do aluno e do professor.
- Spearman, RMSE e MAE do escore do aluno contra o RePO+.
- Custo: 1 passe do aluno versus 100 máscaras do professor.

### R5b — Controle de mascaramento no aluno (recomendado)

O aluno atual é treinado na amostra crua.  O controle treina o mesmo MLP com
**augmentation de mascaramento** (as máscaras que o professor usa).  Isso
separa duas explicações para a perda de robustez: "aluno determinístico" versus
"aluno nunca viu máscara".  É a experiência que transforma o resultado negativo
em diagnóstico.

Precisa de `--student-kind masked_ae` em `distill.py` (hoje só existe o
regressor).

### R6 — Ataque ao aluno (Direção C, adversarial)

```bash
bash cluster/submit.sh mask-exp distill-attack all
```

**Células:** 11 ataques × 5 k; o professor é avaliado nos mesmos exemplos.

**Métricas afetadas:**

- TPR de vítima do aluno no FPR 10 % **dele** (threshold próprio).
- TPR de vítima do professor no threshold do professor, nos mesmos exemplos.
- `fooled` do aluno e do professor; queda condicional de cada um.
- Custo do ataque ao aluno (1 passe por avaliação) versus professor
  (100 máscaras).
- **Produto:** a resposta científica da Direção C — se o aluno mantém o TPR de
  vítima, o ensemble é dispensável; se colapsa, "o ensemble é a defesa".

**Controle de orçamento (recomendado):** comparar o aluno de 1 passe com o
professor de 25 máscaras **no mesmo custo de inferência**.  Sem isso, a queda
do aluno pode ser só a queda de compute, não da destilação.  Exige ou um
ensemble de alunos ou MC-dropout com EOT no ataque (hoje não implementado).

### R7 — Confirmação em pacotes (normal)

```bash
MASK_PACKET_WINDOWS=2000 bash cluster/submit.sh mask-exp packet
```

**Métricas afetadas:** TPR macro/weighted de RePO e RePO+ no FPR 1 % para
`iid-100`, `balanced-25` e `covering-25`; ordenação das políticas.

**Produto:** testa se a conclusão de 25 máscaras é um artefato do modelo de
fluxo.  **Não** testa robustez adversarial de pacotes.

### R8 — Adversarial de pacotes (lacuna)

Não implementado.  O ataque de pacotes do artigo opera em janelas de 96×…
features e é ordens de magnitude mais caro que o de fluxo.  Opções, em ordem de
custo:

1. **Fora de escopo declarado:** manter pacotes como confirmação normal e
   justificar que a evidência adversarial vem do fluxo (é o dataset do artigo).
2. **Ataque reduzido:** 3 ataques × 200–500 janelas, só para mostrar a ordem
   de grandeza.
3. **Completo:** 11 ataques × 2 000 janelas, com laço compilado.

Enquanto R8 não existir, nenhuma afirmação de robustez adversarial em pacotes
pode entrar no relatório.

### R9 — Retreino com a política de máscara (condicional)

O autoencoder é treinado com máscaras `iid` de 75 %.  Se a produção adotar
`balanced`/`covering`, há *train/test mismatch* de distribuição de máscara.
O piloto **não** mostrou queda > 1 pp (G2 não disparou), então o retreino não é
obrigatório.  Ele vira obrigatório se:

- G2 disparar em R1; ou
- a política estruturada for adotada em produção; ou
- um revisor pedir a curva com o modelo retreinado.

Precisa de `--mask-policy` em `train_flow_based_model.py`.

### R10 — Agregação

```bash
bash cluster/submit.sh mask-exp aggregate
```

Faz o merge dos arrays (`mask_experiments.merge`) e gera
`MASK_EFFICIENCY_RESULTS.md` (tabelas por direção, métricas adversariais e
conclusões).  Rodar **depois** de R2, R4, R6 e R7.

## 4. Mapa métrica → runs

| Métrica | Definição | Runs que produzem | Direção |
|---|---|---|---|
| TPR normal macro/weighted @1 % | média por classe no FPR 1 % | R0, R1, R3, R5, R7 | todas |
| TPR de vítima @10 % | TPR no threshold adversarial do próprio detector | R2, R4, R6 | A/B/C |
| Queda condicional | `1 − vítima/normal` entre detectadas antes | R2, R4, R6 | A/B/C |
| `fooled` | evadidas + backup | R2, R4, R6 | A/B/C |
| `no_change` | já benignas antes do ataque | R2, R4 | A/B |
| `bypass` | evadiu o estágio barato | R4 | B |
| Máscaras/amostra (produção) | custo na mistura 99/1 | R1, R3, R5 | A/B/C |
| Máscaras/amostra (ataque) | custo de gerar o adversarial | R2, R4, R6 | A/B/C |
| Spearman / RMSE / MAE | fidelidade ao escore de 100 máscaras | R1, R5, R5b | A/C |
| Concordância com RePO+ | fração de decisões iguais | R3 | B |
| TPR em pacotes | confirmação em outro modelo | R7 | A (confirmação) |

## 5. Ordem e dependências

```
R0 baseline
 └─ R1 variance
     └─ R2 adversarial all
R3 cascade
 └─ R4 cascade-attack all
R5 distill
 └─ R6 distill-attack all
R7 packet
R8 (opcional / fora de escopo)
R9 (só se G2 falhar)
R10 aggregate  ← depende de R2, R4, R6, R7
```

Encadeamento no Apuana:

```bash
bash cluster/run_mask_study.sh            # matriz completa
bash cluster/run_mask_study.sh --smoke    # smoke de todas as etapas
bash cluster/run_mask_study.sh --dry-run  # só imprime o plano
```

O script submete todas as etapas com `afterok`, grava o manifesto
`$RESULTS_ROOT/mask_efficiency/chain_manifest.tsv` e injeta
`MASK_STEP_NOTE` em cada job (o log começa com `[step] <descrição>`).  Se
preferir encadear manualmente:

```bash
BASE=$(bash cluster/submit.sh mask-exp baseline | awk 'NR==1{print $NF}')
VAR=$(DEPENDENCY="afterok:${BASE}" bash cluster/submit.sh mask-exp variance | awk 'NR==1{print $NF}')
ADV=$(DEPENDENCY="afterok:${VAR}" bash cluster/submit.sh mask-exp adversarial all | awk 'NR==1{print $NF}')

CAS=$(bash cluster/submit.sh mask-exp cascade | awk 'NR==1{print $NF}')
CASATK=$(DEPENDENCY="afterok:${CAS}" bash cluster/submit.sh mask-exp cascade-attack all | awk 'NR==1{print $NF}')

DIS=$(bash cluster/submit.sh mask-exp distill | awk 'NR==1{print $NF}')
DISATK=$(DEPENDENCY="afterok:${DIS}" bash cluster/submit.sh mask-exp distill-attack all | awk 'NR==1{print $NF}')

PKT=$(bash cluster/submit.sh mask-exp packet | awk 'NR==1{print $NF}')

DEPENDENCY="afterok:${ADV}:${CASATK}:${DISATK}:${PKT}" \
  bash cluster/submit.sh mask-exp aggregate
```

O `aggregate` só roda se todos os arrays terminarem com sucesso; se um ataque
falhar, reenvie só aquele (`mask-exp adversarial <ataque>`) e depois o
`aggregate`.

## 6. Lacunas de implementação (o que impede a matriz completa)

| Lacuna | Impacto | Esforço |
|---|---|---|
| **Subdiretório por estágio adversarial** | permite curva em 4 ataques + finalistas em 11 sem sobrescrever JSON | pequeno: `--out-dir`/`MASK_ADV_TAG` + merge recursivo |
| **`--student-kind masked_ae`** | controle que separa "determinístico" de "sem máscara" | médio: augmentation + treino |
| **Aluno classificador (cabeça binária)** | terceiro aluno do plano | médio |
| **Orçamento de ataque equalizado** | comparar 25 vs 100 máscaras no mesmo nº de forward passes | médio: cap de `masks_evaluated` no `FlowAttack` |
| **Aluno compute-matched** | atribuir a perda a destilação vs compute | médio/alto: ensemble de alunos ou MC-dropout + EOT |
| **Ataque de pacotes** | afirmação adversarial em pacotes | alto |
| **`--mask-policy` no treino** | retreino condicional (G2) | médio |
| **`shared` / `fixed` mask policy** | teste de Kerckhoffs e máscara fixa de implantação | pequeno/médio |

As três primeiras são as que mais mudam a conclusão científica; as demais são
controles ou confirmações.

## 7. Mínimo publicável × matriz completa

**Mínimo publicável (se o orçamento de GPU apertar):**

- R0, R1 (5 rep.), R2 com 5 variantes (`iid-100/25/10`, `balanced-25`,
  `covering-25`) nos 11 ataques, R3, R4 com `cheap=25`, R5, R6 nos 11 ataques,
  R7 e R10.
- Resultado: curva custo × TPR normal e custo × TPR de vítima; cascata com
  bypass; aluno com e sem ataque; confirmação normal em pacotes.

**Matriz completa:**

- R2 com 9 variantes, R4 com 3 estágios, R5b, R6 + aluno compute-matched, R8
  (se viável), R9 se G2 disparar.
- Resultado: Pareto por política, sensibilidade do `t_low`, diagnóstico da
  destilação e confirmação adversarial em pacotes.

## 8. Critérios de decisão (o que cada resultado implica)

| Gate | Condição | Decisão |
|---|---|---|
| G1 | baseline não reproduz | parar e corrigir antes de tudo |
| G2 | `balanced`/`covering` caem > 1 pp no TPR normal | rodar R9 antes de descartar |
| G3 | bypass do estágio barato alto | cascata é resultado negativo, não recomendação |
| G4 | TPR de vítima do aluno cai vs professor | destilação não substitui o ensemble sob ataque |
| G5 | ordenação de políticas muda em pacotes | conclusão de A é específica do fluxo |

## 9. Estimativa de custo (ordem de grandeza)

Ancorada no piloto local (CPU, ~2 s por amostra de ataque de fluxo) e nos
tempos medidos no Apuana em `cluster/README.md` (15 min–2 h por ataque com 100
máscaras na A100).  **Calibrar com um smoke de 1 ataque × 200 amostras antes
de submeter a matriz inteira.**

| Run | Células de ataque | Custo relativo | Observação |
|---|---|---|---|
| R2 (9 variantes) | ~495 k | **dominante** | ≈ 9× uma variante de 100 máscaras por ataque |
| R4 (3 estágios) | ~165 k | ~1/3 de R2 | ataque ao prefixo é mais barato |
| R6 (1 aluno) | ~55 k | ~1/9 de R2 | aluno é 1 passe |
| R1 | 90 passagens normais | baixo | sem ataque |
| R3 | 9 configs × 5 rep. | baixo | sem ataque |
| R5 | treino + escore do professor | baixo | 40 épocas |
| R7 | 2 000 janelas × 3 | baixo/médio | modelo de pacotes |

Com `CONCURRENCY=2` e 11 jobs por array, a parede é dominada por R2; o
`aggregate` roda em minutos.

## 10. O que já está pronto no piloto

- R0: reproduziu thresholds e TPR de referência.
- R1: 20–25 máscaras dentro de 1 pp do TPR de 100, com ~4–5× menos linhas.
- R2: 4 variantes × 4 ataques × 20 amostras — **só orienta**, não conclui.
- R3: sequencial com 5,6 máscaras/amostra e 99,7 % de concordância; cascata
  economiza 30–43 % dos benignos.
- R4: sem bypass do estágio barato no piloto, com TPR de vítima igual ao normal.
- R5/R6: aluno reproduz o professor no normal (Spearman 0,948), mas perde
  ~29 % do TPR de vítima contra ~9 % do professor.
- R7: ordenação `iid-100 > balanced-25 > covering-25` se mantém em pacotes.

O que o piloto **não** permite afirmar: qual variante de A é mais robusta
(20 amostras), se a cascata é segura (só 2 estágios × 4 ataques), e se a perda
do aluno é destilação ou compute (falta R5b e o controle equalizado).
