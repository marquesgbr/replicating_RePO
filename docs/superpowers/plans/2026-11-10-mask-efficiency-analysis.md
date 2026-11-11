# Análise completa do sistema de máscaras do RePO — Plano de pesquisa

> **Para quem executa:** este plano é um experimento de pesquisa, não um
> refactor. Use `superpowers:executing-plans` (inline) ou
> `superpowers:subagent-driven-development` (subagentes) e marque os
> checkboxes (`- [ ]`) conforme avança.

**Goal:** caracterizar e deslocar a fronteira **custo × robustez** do RePO+,
avaliando três direções — (A) menos máscaras / máscaras estruturadas,
(B) decisão em cascata/sequencial, (C) destilação do ensemble — com
recalibração de threshold e **ataques adaptativos** em todas as variantes, de
forma que os resultados sejam publicáveis e não apenas microbenchmarks.

**Architecture:** o protocolo atual de `code/new_data_eval/` já é determinístico
por amostra e serve de base. Introduzimos uma abstração de `MaskPolicy`
(como as máscaras são geradas) e de `ScorePolicy` (como os erros viram escore
e decisão), mantendo o comportamento padrão **bit-idêntico** ao atual. O ataque
passa a receber a política do detector para poder ser adaptativo. Um runner de
experimentos separado gera JSONs de resultado com métricas de custo e
robustez; a agregação produz as curvas e tabelas do relatório.

**Tech Stack:** Python 3.10+, TensorFlow 2.21/Keras 3, NumPy, pytest, SLURM
(Apuana), matplotlib para figuras, LaTeX para o relatório.

**Branch:** `feat/mask-efficiency` (criada de `feat/apuana-cluster`).

**Baseline de referência:** `docs/REPLICATION_ANALYSIS.md` e os resultados
commitados em `results/` (fluxo e pacotes, 100 máscaras, seed 2019).

---

## 1. Intenção e enquadramento

O RePO usa 100 máscaras por amostra porque o escore é um ensemble de erros de
reconstrução. Em produção isso é 100× o custo de um autoencoder de passe único.
A pergunta científica não é "dá para usar menos máscaras?" — quase certamente
dá — mas **quanto de robustez adversarial se perde em cada ponto da curva e
qual é o desenho de máscara que compra mais robustez por forward pass**.

Três resultados são possíveis e todos são publicáveis:

1. **Positivo:** existe uma política de máscara com custo bem menor e perda de
   robustez desprezível (ex.: máscaras balanceadas + 25 máscaras).
2. **Tradeoff:** a curva custo × robustez é íngreme e quantificável; o artigo
   original não a reporta.
3. **Negativo:** qualquer redução de máscaras (ou destilação) colapsa sob
   ataque adaptativo; o ensemble aleatório de 100 máscaras é necessário e agora
   há evidência.

## 2. Objetivos

- Medir a **variância** dos escores RePO e RePO+ em função do número e do
  desenho das máscaras.
- Medir **TPR normal e TPR de vítima** em função de `n_masks`, com threshold
  recalibrado para o mesmo FPR.
- Avaliar **máscaras balanceadas** (exatamente `k` features preservadas) e
  **máscaras com cobertura garantida por grupo** como redutores de variância.
- Avaliar uma **cascata** (estágio barato + ensemble só na região duvidosa) e um
  **escore sequencial** com parada antecipada, incluindo o custo médio em um
  fluxo benigno-dominante.
- Avaliar **destilação** do escore RePO+ em um aluno de passe único.
- Em todas as variantes, rodar **(a)** o ataque de referência e **(b)** um
  ataque adaptativo que conhece a política do detector.
- Produzir tabelas e figuras prontas para o relatório, com intervalos de
  confiança e custo computacional medido.

## 3. Não objetivos

- **Direção D (tuning de engenharia)** fica fora: fp16, poda e redução de
  largura não são contribuição científica. Só entram como *configuração fixa*
  do ambiente, se necessário para caber no orçamento.
- Não alterar os notebooks originais nem os resultados já commitados.
- Não mudar o protocolo `primary` (100 máscaras, 75%, seed 2019) — ele é o
  controle.
- Não retreinar os modelos de referência, exceto se uma variante de máscara
  criar *train/test mismatch* mensurável (ver Task 1, gate G2).
- Não usar o novo dataset como resultado principal; no máximo como validação
  externa de um finalista.

## 4. Invariantes do protocolo de avaliação

Toda variante segue estas regras, sem exceção:

1. **FPR casado.** A comparação principal é com thresholds recalibrados no
   mesmo FPR (1% normal, 10% adversarial). O threshold do artigo também é
   reportado, mas nunca usado para declarar ganho.
2. **Ataque adaptativo obrigatório.** O atacante conhece `n_masks`, o
   agrupamento, a estatística (RePO/RePO+), a regra de decisão e a política de
   máscara. Se a política for determinística, o atacante a conhece exatamente.
3. **Orçamento de ataque normalizado.** Reportar duas comparações: mesmo número
   de iterações e mesmo número de forward passes. A segunda é a honesta para
   "custo × robustez".
4. **Estocasticidade controlada.** `score_repeats ≥ 5` no estudo de variância e
   ≥ 3 na avaliação adversarial; reportar média e IC bootstrap sobre amostras e
   repetições. Nunca comparar uma passada contra outra.
5. **Mesmo conjunto de teste.** Amostragem aleatória com seed fixa, não as
   primeiras N (corrige o viés documentado).
6. **Registro de custo.** Todo JSON grava linhas mascaradas, chamadas ao
   modelo, tempo de parede e `config_hash`.

## 5. Abstrações a introduzir

### 5.1 `MaskPolicy`

Novo módulo `code/new_data_eval/masking.py`:

```python
class MaskPolicy(Protocol):
    name: str
    def masks(self, *, seed: int, sample_index: int, phase: str,
              repeat: int, n_masks: int, n_features: int) -> np.ndarray: ...
```

Implementações:

| Política | Descrição | Usa RNG? |
|---|---|---|
| `IIDMaskPolicy` | comportamento atual: `uniform > 0.75` | sim, stateless |
| `BalancedMaskPolicy` | cada máscara preserva exatamente `k = round(0.25·F)` features | sim, stateless |
| `CoveringMaskPolicy` | balanceada + garante que cada feature aparece ao menos uma vez no grupo de `n_masks//5` | sim, com correção determinística |
| `SharedMaskPolicy` | um conjunto de máscaras por batch, reutilizado entre amostras | sim |
| `FixedMaskPolicy` | conjunto fixo por seed de implantação (para o teste de Kerckhoffs) | sim |

Requisitos: shape `(n_masks, F)`, dtype float32, determinismo por
`(seed, sample_index, phase, repeat)`, testes de propriedade (contagem exata,
cobertura, independência entre fases).

### 5.2 `ScorePolicy`

Novo módulo `code/new_data_eval/decision.py`:

```python
@dataclass
class Decision:
    score: float
    masks_used: int
    model_calls: int
    routed_to_full: bool

class ScorePolicy(Protocol):
    def decide(self, errors: np.ndarray) -> Decision: ...
```

Implementações: `RePO`, `RePOPlus`, `Cascade(cheap_masks, t_low, full)`,
`SequentialRePO(ci_half_width)`. O `Cascade` recebe os erros do estágio barato
e, se necessário, pede o restante ao chamador (uma função `score_range`), para
não carregar o modelo dentro da política.

### 5.3 Adaptação do ataque

`code/new_data_eval/attack.py` ganha:

- `detector_n_masks` e `detector_policy` — o atacante usa a mesma estatística e
  o mesmo número de máscaras do detector;
- `mask_source` — `fresh_per_iteration` (atual), `reuse_iteration_masks`,
  `detector_masks` (ataque adaptativo a política fixa);
- contador de forward passes no `AttackOutcome`.

O baseline com `detector_n_masks=100` e `fresh_per_iteration` deve reproduzir o
ataque atual (teste de regressão).

---

## 6. Matriz de experimentos

### 6.1 Estudo de variância (Task 1)

| Fator | Valores |
|---|---|
| Dataset | fluxo (CICIDS), 100 k benignos aleatórios + até 5 k por ataque |
| `n_masks` | 5, 10, 20, 25, 50, 100 |
| Política | `iid`, `balanced`, `covering`, `shared` |
| Repetições | 5 conjuntos independentes |

Métricas: média/desvio do escore por amostra, correlação de Spearman com o
escore de 100 máscaras, TPR por classe no FPR casado, meia-largura do IC do TPR,
custo em linhas mascaradas.

### 6.2 Avaliação adversarial (Task 2)

Ataques representativos para iterar: **FTP-Patator** (ataque mais eficaz),
**Slowloris/Slowhttptest** (custo médio), **DoS Hulk** e **DDoS** (TPR alto e
muitas amostras). Os 11 ataques só no resultado final.

| Variante | `n_masks` | Política | Ataque |
|---|---|---|---|
| Controle | 100 | iid | referência |
| A1 | 25 | iid | referência |
| A2 | 25 | iid | adaptativo (25) |
| A3 | 25 | balanced | adaptativo (25) |
| A4 | 50 | balanced | adaptativo (50) |
| A5 | 25 | covering | adaptativo (25) |

Métricas: TPR de vítima no FPR 10%, queda condicional entre originalmente
detectados, taxa de `fooled`, forward passes por amostra, tempo.

### 6.3 Cascata e sequencial (Task 3)

- Cascata: estágio barato com `k ∈ {1, 4}` máscaras e `t_low` calibrado para
  FN ≈ 0 na validação; ensemble completo só acima de `t_low`.
- Sequencial: para RePO, parada quando o IC da média exclui o threshold;
  para RePO+, avaliar em blocos de 5 grupos com regra conservadora.
- Ataque adaptativo **no estágio barato** (o elo fraco) e no ensemble.
- Métricas: máscaras médias por amostra em mistura realista (99% benigno),
  throughput, TPR normal e de vítima, taxa de bypass do estágio barato.

### 6.4 Destilação (Task 4)

- Aluno 1: regressão do escore RePO+ a partir da amostra crua.
- Aluno 2: cabeça de decisão binária benigno/anômalo.
- Aluno 3: mesmo autoencoder com augmentation de mascaramento (controle).
- Avaliação: TPR no FPR casado; ataque adaptativo direto no aluno; comparação
  com o professor. Se a robustez colapsar, o resultado negativo é o produto.

### 6.5 Confirmação em pacotes (Task 5)

Somente o controle + os dois melhores finalistas, nos 11 ataques, com o laço
compilado (`PACKET_GPU_LOOP=1`).

---

## 7. Orçamento computacional (A100, ordem de grandeza)

| Etapa | Custo estimado |
|---|---|
| Task 0 (baseline + instrumentação) | ~2 h GPU |
| Task 1 (variância, 4 políticas × 6 n) | ~2–4 h GPU |
| Task 2 (4 ataques × 6 variantes × 5 k) | ~1–3 dias GPU (2 concorrentes) |
| Task 3 (cascata + sequencial + ataques) | ~1–2 dias GPU |
| Task 4 (destilação: 3 alunos + ataques) | ~2–4 dias GPU |
| Task 5 (pacotes, 3 variantes × 11 ataques) | ~2–4 dias GPU |
| Task 6 (agregação/figuras) | ~1 dia CPU |

Os ataques dominam; por isso a iteração usa 4 ataques representativos e só o
resultado final roda os 11.

---

## 8. Estrutura de arquivos

**Novos:**

- `code/new_data_eval/masking.py` — políticas de máscara.
- `code/new_data_eval/decision.py` — políticas de escore/decisão.
- `code/mask_experiments/__init__.py`
- `code/mask_experiments/runner.py` — executa uma célula da matriz e grava JSON.
- `code/mask_experiments/aggregate.py` — consolida e gera CSV/figuras.
- `code/mask_experiments/analyze_variance.py` — curvas de variância e TPR.
- `code/mask_experiments/tests/` — testes das políticas e da agregação.
- `docs/superpowers/specs/2026-11-10-mask-efficiency-design.md` — design
  detalhado (gerado a partir deste plano antes da implementação).
- `results/mask_efficiency/` — saída (não versionar dados grandes).

**Modificados:**

- `code/new_data_eval/scoring.py` — delega a `MaskPolicy`; default idêntico.
- `code/new_data_eval/attack.py` — `detector_n_masks`, `mask_source`, custo.
- `code/new_data_eval/cli.py` — flags `--mask-policy`, `--n-masks`,
  `--cascade`, `--score-repeats` (já existe).
- `code/new_data_eval/metrics.py` — campos de custo no JSON.
- `cluster/metrics.py` e `cluster/repo_runner.py` — apenas o necessário para o
  runner de cluster chamar as novas políticas (sem mudar o default).
- `cluster/submit.sh` / `cluster/slurm/job.sh` — namespace `mask-exp`.
- `report.tex` / `docs/` — seção de resultados.

---

## 9. Tarefas

### Task 0: Baseline congelado e instrumentação

**Files:**
- Modify: `code/new_data_eval/metrics.py`, `code/new_data_eval/scoring.py`
- Create: `code/mask_experiments/runner.py` (esqueleto), `code/mask_experiments/tests/test_baseline_equivalence.py`

**Interfaces:**
- Consumes: pipeline atual (`score_samples`, `reconstruction_errors`).
- Produces: JSON de baseline com custo; garantia de equivalência bit-a-bit.

- [x] **Step 1:** rodar o baseline de fluxo (100 máscaras, seed 2019, mesmos
  dados) e arquivar como `results/mask_efficiency/baseline_flow.json`.
  *(feito como smoke CPU: 10 k benignos + 500 por ataque, arquivado em
  `results/mask_efficiency/baseline_flow_smoke.json`; a rodada completa de
  100 k/5 k fica para o cluster)*
- [x] **Step 2:** adicionar contadores de linhas mascaradas, chamadas ao modelo
  e tempo de parede ao JSON.
- [x] **Step 3:** escrever teste que compara os escores do baseline antigo e do
  novo com tolerância zero (ou `1e-6` documentado). *(golden de máscaras,
  golden numérico dos escores e comparação bit-a-bit contra o `scoring.py`
  anterior)*
- [x] **Step 4:** rodar `pytest code/new_data_eval/tests cluster/tests`.
  *(227 testes de `new_data_eval` + 42 do cluster + 8 de `mask_experiments`
  passando na venv com TensorFlow 2.21 CPU)*

**Verificação:** baseline reproduz os TPRs de `REPLICATION_ANALYSIS.md` dentro
do ruído; teste de equivalência verde.

**Resultado (CPU, 2026-11-10):** com 10 k benignos + 500 amostras por ataque
(15.047 amostras), o smoke reproduziu os thresholds de referência e o TPR macro:

| Modo | Threshold (FPR 1%) | Ref. | TPR macro | Ref. |
|---|---|---|---|---|
| RePO | 0,04109 | 0,04130 | 0,2407 | 0,2443 |
| RePO+ | 0,10578 | 0,10608 | 0,2580 | 0,2525 |

Custo medido: 3,01 M linhas mascaradas, 30.094 chamadas ao modelo, 35,3 s de
modelo (1,17 ms por chamada de 100 linhas). O baseline completo (100 k benignos
+ 5 k por ataque) roda no Apuana com o mesmo comando e grava
`results/mask_efficiency/baseline_flow.json`.

**Gate G1:** se o baseline não reproduzir, parar e corrigir antes de qualquer
variante.

---

### Task 1: Políticas de máscara e estudo de variância (Direção A, parte 1)

**Files:**
- Create: `code/new_data_eval/masking.py`, `code/mask_experiments/analyze_variance.py`
- Create: `code/mask_experiments/tests/test_masking.py`
- Modify: `code/new_data_eval/scoring.py` (injeção da política)

**Interfaces:**
- Produces: `MaskPolicy.masks(...) -> np.ndarray (n_masks, F)`.
- Consumes: `scoring._sample_masks` passa a delegar à política.

- [ ] **Step 1:** implementar `IIDMaskPolicy` com saída idêntica à atual.
- [ ] **Step 2:** testes de shape, dtype, determinismo e independência de fase.
- [ ] **Step 3:** implementar `BalancedMaskPolicy` (`k = round(0.25·F)`) e
  `CoveringMaskPolicy`; testes de contagem/cobertura.
- [ ] **Step 4:** rodar a matriz 6.1 e gerar:
  - curva TPR × `n_masks` por política;
  - desvio do escore por amostra;
  - Spearman contra o escore de 100 máscaras;
  - custo (linhas mascaradas).
- [ ] **Step 5:** escrever `results/mask_efficiency/variance_summary.json` e
  figuras em `docs/figures/` (ou diretório de resultados).

**Verificação:** a curva mostra o "joelho" de custo; ICs reportados.

**Gate G2:** se `balanced`/`covering` degradarem o TPR normal em > 1 pp,
testar retreino com a mesma política de máscara antes de descartar (é o único
caso que autoriza retreinar).

---

### Task 2: Avaliação adversarial adaptativa (Direção A, parte 2)

**Files:**
- Modify: `code/new_data_eval/attack.py`, `code/new_data_eval/cli.py`
- Create: `code/mask_experiments/runner.py` (modo adversarial)
- Create: `code/new_data_eval/tests/test_attack_adaptive.py`

**Interfaces:**
- `FlowAttack(..., detector_n_masks, mask_source)`; `AttackOutcome` ganha
  `model_calls` e `masks_used`.
- O ataque de referência com os defaults não muda (teste de regressão).

- [ ] **Step 1:** implementar `detector_n_masks` e `mask_source`.
- [ ] **Step 2:** teste de regressão: defaults produzem o mesmo `fooled` e
  `victim_tpr` do baseline em 200 amostras.
- [ ] **Step 3:** rodar a matriz 6.2 nos 4 ataques representativos.
- [ ] **Step 4:** repetir com orçamento de forward passes igualado.
- [ ] **Step 5:** gerar a figura custo × TPR de vítima com IC.

**Verificação:** toda variante tem resultado com ataque adaptativo; nenhum
ganho é declarado com o ataque antigo apenas.

---

### Task 3: Cascata e decisão sequencial (Direção B)

**Files:**
- Create: `code/new_data_eval/decision.py`
- Create: `code/mask_experiments/tests/test_cascade.py`
- Modify: `code/new_data_eval/cli.py`, `code/new_data_eval/scoring.py`

**Interfaces:**
- `Cascade(cheap_masks=1|4, t_low, full_n_masks)` retorna `Decision` com
  `masks_used` e `routed_to_full`.
- `t_low` é calibrado em validação para FN ≈ 0 (definir o alvo exato no design).

- [ ] **Step 1:** implementar a cascata e o cálculo de `t_low` na validação.
- [ ] **Step 2:** medir máscaras médias por amostra em mistura 99% benigno.
- [ ] **Step 3:** medir TPR normal e vítima no FPR casado.
- [ ] **Step 4:** ataque adaptativo otimizando **o estágio barato**; medir
  bypass e custo do ataque.
- [ ] **Step 5:** implementar `SequentialRePO` com IC e comparar com a cascata.
- [ ] **Step 6:** decidir se a cascata entra no resultado final.

**Verificação:** relatório mostra o ganho médio e o pior caso de segurança.

**Gate G3:** se o ataque adaptativo derrubar o estágio barato com facilidade, a
cascata vira resultado negativo documentado, não recomendação.

---

### Task 4: Destilação do ensemble (Direção C)

**Files:**
- Create: `code/mask_experiments/distill.py`
- Create: `code/mask_experiments/tests/test_distill.py`
- Modify: `report.tex` (seção de resultados negativos/positivos)

**Interfaces:**
- `train_student(kind ∈ {regressor, classifier, masked_ae}, data, teacher)`.
- Avaliação usa o mesmo ataque adaptativo, apontado ao aluno.

- [ ] **Step 1:** gerar o alvo (escore RePO+ do professor) para treino/val.
- [ ] **Step 2:** treinar os três alunos com a mesma arquitetura base e seed.
- [ ] **Step 3:** avaliar TPR normal no FPR casado.
- [ ] **Step 4:** ataque adaptativo direto no aluno; comparar com o professor.
- [ ] **Step 5:** analisar: se a robustez cair, documentar como resultado
  negativo ("o ensemble não é destilável"); se não, investigar por quê.

**Verificação:** conclusão sustentada por ataque adaptativo, não apenas por
TPR normal.

---

### Task 5: Confirmação em pacotes

**Files:**
- Modify: `cluster/submit.sh`, `cluster/slurm/job.sh` (namespace `mask-exp`)
- Create: `results/mask_efficiency/packet_*.json`

- [ ] **Step 1:** rodar controle + 2 finalistas nos 11 ataques, com
  `PACKET_GPU_LOOP=1`.
- [ ] **Step 2:** verificar se a ordenação das variantes se mantém.
- [ ] **Step 3:** registrar custo por pacote.

**Verificação:** conclusões do fluxo não são contrariadas nos pacotes (ou a
contradição é reportada).

---

### Task 6: Agregação e redação

**Files:**
- Create: `code/mask_experiments/aggregate.py`
- Create: `docs/MASK_EFFICIENCY_RUNS.md` (matriz de runs e custo)
- Modify: `report.tex`, `docs/REPLICATION_ANALYSIS.md` (seção nova),
  `docs/NEW_DATASET.md` (nota de protocolo, se aplicável)
- Create: `docs/figures/cost_robustness.pdf`, `docs/figures/n_masks_curve.pdf`

- [ ] **Step 1:** consolidar JSONs em CSV com IC bootstrap.
- [ ] **Step 2:** gerar as figuras custo × robustez e TPR × `n_masks`.
- [ ] **Step 3:** escrever a seção do relatório com: motivação, método,
  resultados, limitações e reprodutibilidade.
- [ ] **Step 4:** rodar a suíte completa e o smoke de cada variante.
- [ ] **Step 5:** revisar com `superpowers:requesting-code-review` e
  `verification-before-completion` antes de fechar a branch.

**Verificação:** qualquer número no texto é rastreável a um JSON com
`config_hash`, seed e comando.

---

## 10. Critérios de aceitação

1. O baseline é reproduzido antes de qualquer variante (G1).
2. Toda variante tem avaliação normal **e** adversarial adaptativa.
3. Toda comparação usa FPR casado e IC.
4. A curva custo × robustez existe para fluxo e, nos finalistas, para pacotes.
5. O custo é medido (linhas mascaradas, chamadas, tempo), não estimado.
6. Resultados negativos são relatados com o mesmo rigor que os positivos.
7. Nenhum artefato grande (CSV/NPZ/checkpoint) entra no git.
8. `pytest` verde e smoke de cada variante documentado.

## 11. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Reduzir máscaras enfraquece o RePO+ e o ganho some | medir a curva antes de escolher; reportar o tradeoff |
| Ataque adaptativo é caro | 4 ataques na iteração, 11 só no final; orçamento igualado |
| Política determinística quebra sob Kerckhoffs | `FixedMaskPolicy` só como teste de estresse explícito |
| Balanceada cria mismatch com o modelo treinado | gate G2; retreino só se necessário |
| Destilação colapsa | é um resultado aceitável e esperado; documentar |
| Cascata dá falsa sensação de segurança | ataque no estágio barato é obrigatório (G3) |
| Ruído estocástico confunde efeito | `score_repeats ≥ 3-5` e IC em tudo |

## 12. Decisões em aberto (defaults propostos)

- **Dataset principal:** fluxo CICIDS; pacotes só para confirmação. *(default)*
- **FPR casado:** 1% normal, 10% adversarial. *(default)*
- **`n_masks` candidatos:** múltiplos de 5 até 100. *(default)*
- **Ataques de iteração:** FTP-Patator, Slowhttptest, DoS Hulk, DDoS. *(default)*
- **Alvo de FN da cascata:** 0,1% dos ataques na validação. *(default)*
- **Arquitetura dos alunos:** mesma `FlowModel`. *(default)*
- **Retreino:** só no gate G2. *(default)*

Se algum default não servir, o plano muda no design detalhado antes da
implementação.

## 13. Ordem de execução

```
Task 0 (baseline) ──► Task 1 (variância) ──► G2
                                   │
                                   ├──► Task 2 (adversarial A)
                                   ├──► Task 3 (cascata B)
                                   └──► Task 4 (destilação C)
                                                │
                          finalistas ───────────┴──► Task 5 (pacotes)
                                                          │
                                                          ▼
                                                     Task 6 (redação)
```

Tasks 2, 3 e 4 são independentes depois de Task 1 e podem rodar em paralelo
(2 jobs concorrentes no Apuana).

## 14. Status de execução (piloto em CPU, 2026-11-10)

Executado na branch `feat/mask-efficiency`, em CPU (TensorFlow 2.21), com o
harness de `code/mask_experiments/`:

- **Task 0:** concluída; baseline de 15.047 amostras reproduziu threshold e
  TPR macro de referência (ver seção 9 da Task 0).
- **Task 1:** concluída no piloto (5 k benignos + 300 por ataque, 3 repetições,
  3 políticas × 6 valores de `n_masks`); resultados em
  `results/mask_efficiency/variance_flow.json`.
- **Task 2:** concluída no piloto (5 variantes × 4 ataques × 20 amostras,
  ataque adaptativo); resultados em `adversarial_pilot.json`.
- **Task 3:** concluída no piloto (cascata e RePO sequencial); resultados em
  `cascade_flow.json`. O ataque **direto** no estágio barato foi executado em
  `cascade_attack.json` (o ataque de referência não conseguiu bypassar
  `t_low`).
- **Task 4:** concluída no piloto; aluno de passe único treinado, avaliado no
  cenário normal (`distillation_flow.json`) e atacado diretamente
  (`distill_attack.json`). A destilação perde robustez adversarial.
- **Task 5:** parcial; avaliação **normal** dos pacotes com as três políticas
  (`packet_normal.json`). O ataque de pacotes não foi executado.
- **Task 6:** consolidado em `docs/MASK_EFFICIENCY_RESULTS.md`; a seção do
  `report.tex` fica pendente até a rodada completa.

**Integração com o Apuana (2026-11-10):** o namespace `mask-exp` de
`cluster/submit.sh` e os casos `mask_*` de `cluster/slurm/job.sh` executam
todas as direções, com arrays por ataque, merge automático
(`mask_experiments.merge`) e relatório consolidado
(`mask_experiments.aggregate`).  A cadeia completa é submetida por
`cluster/run_mask_study.sh` (dependências `afterok`, manifesto em
`chain_manifest.tsv` e `[step]` no log de cada job).  Ver
`cluster/README.md`, seção 17, e `docs/APUANA_RUNBOOK.md`, seção 10.1.  As
métricas adversariais por direção estão na seção 2 do relatório.

As pendências (arrays completos de 11 ataques, ataque de pacotes, retreino
com máscaras estruturadas e rodada em GPU) entram na execução do Apuana.


