# Avaliação do RePO/RePO+ no novo dataset (CIC-ToN-IoT) no cluster Apuana

> Documento gerado a partir da execução no cluster Apuana (CIn/UFPE).
> Consolida a avaliação normal completa e a rodada adversarial dos nove ataques
> no dataset adicional. Os resultados de fluxo e pacotes do artigo estão em
> `docs/REPLICATION_ANALYSIS.md`; a comparação entre os três conjuntos está na
> seção 5 deste documento.

## 1. Resumo executivo

- O protocolo RePO/RePO+ foi aplicado ao dataset adicional (`data/new_data/`)
  com o modelo pré-treinado dos autores, 100 máscaras por amostra e 75% das
  features mascaradas, sem alterar o fluxo de avaliação do artigo.
- A avaliação **normal** percorreu **3.590.733 amostras válidas** de teste em
  **45,8 min** e calibrou os thresholds nos 754.209 benignos.
- No **FPR de 10%**, o RePO+ obteve **TPR macro 0,6349** e **ponderado 0,7700**,
  contra **0,4914** e **0,5012** do RePO (+14,3 e +26,9 pontos percentuais).
  No **FPR de 1%**, o RePO+ ficou em **0,2013** macro contra **0,1384** do RePO.
  A direção do ganho reproduz a do artigo.
- O desempenho é muito desigual por classe: backdoor (0,9957), password
  (0,9697), injection (0,9221) e ransomware (0,9176) são bem detectados; xss,
  que concentra 60% do teste, fica em **0,7171**; ddos (0,0050) e dos (0,0000)
  praticamente não são detectados nesse threshold.
- A rodada **adversarial** avaliou **30.864 amostras** nos nove ataques, em
  **31,4 h de parede** (62,5 h somadas de GPU).
- **O ataque é praticamente ineficaz neste dataset**: apenas **3 evasões em
  30.864 amostras (0,01%)**, todas em `ransomware`. Nenhum ataque com TPR
  normal relevante reduziu a detecção em mais de 0,2 ponto percentual, e a maior
  queda condicional foi de **0,39%** (`mitm`).
- Em **7 dos 9 ataques** o TPR de vítima continuou acima do FPR de 10%. Os dois
  restantes (`ddos` e `dos`) já tinham TPR normal ≈ 0, então não há o que
  derrubar — é uma limitação da métrica, não um sucesso do ataque.
- Na comparação com os dados do artigo, o contraste é grande: no fluxo do
  CICIDS2017 o FTP-Patator derrubou o TPR em 40,7 pontos percentuais (82% entre
  as amostras detectadas); aqui a pior queda foi de 0,39%. A robustez do RePO+
  é, portanto, **dependente do dataset**.

## 2. Dataset e método

### 2.1 Dados

O ZIP canônico é `data/new_data/New-dataset-split-clean.zip` (Git LFS), com os
CSVs já processados e divididos. O dataset é atribuído ao **CIC-ToN-IoT** pelo
nome do arquivo, pelo script legado e pelo cabeçalho dos CSVs; a citação
oficial ainda está **pendente de confirmação** (`docs/NEW_DATASET.md`, seção
“Fonte e citação”).

| Item | Valor |
| --- | --- |
| Treino (`new_train.csv`) | 1.759.850 linhas, somente `BENIGN` |
| Teste (`new_test.csv`) | 3.591.095 linhas |
| Linhas descartadas (`inf`/`nan`) | 362 |
| **Teste válido** | **3.590.733** |
| Benignos no teste | 754.209 |
| Features do modelo | 77 (`Dst Port` + 76 features de fluxo) |
| Normalização | min–max com `train_min`/`train_max` de `data/new_data/x_train_meta/` |

| Classe | Amostras de teste |
| --- | ---: |
| `BENIGN` | 754.209 |
| `xss` | 2.149.308 |
| `password` | 340.208 |
| `injection` | 277.696 |
| `scanning` | 36.205 |
| `backdoor` | 27.145 |
| `ransomware` | 5.098 |
| `mitm` | 517 |
| `ddos` | 202 |
| `dos` | 145 |

O split separa apenas benignos: 70% para treino e 30% para teste, mais todas as
amostras maliciosas no teste. Não há conjunto de validação separado, e o
threshold é calibrado em benignos que não foram vistos no treino.

### 2.2 Protocolo

Preservado do artigo, sem alterações:

- **100 máscaras** por amostra, cada feature mascarada com probabilidade
  **0,75**;
- **RePO** = média do erro de reconstrução das 100 máscaras;
- **RePO+** = erro dividido em 5 grupos de 20 máscaras, mínimo de cada grupo,
  soma dos 5 mínimos;
- máscaras derivadas deterministicamente de `(seed, sample_index, fase,
  repetição)` com `seed = 2019`;
- thresholds calculados nos escores benignos para **FPR 1%** e **FPR 10%**;
- ataque adversarial de fluxo com 540 iterações por amostra (3 máscaras × 4
  learning rates × 180 iterações), otimizando a perda RePO;
- sucesso medido com o agregado **RePO+**, usando o threshold de FPR 10%.

### 2.3 Ambiente

| Item | Valor |
| --- | --- |
| Cluster | Apuana (CIn/UFPE), nós `cluster-node6` e `cluster-node9` |
| GPU | NVIDIA A100 80 GB PCIe (parte das tarefas na fatia MIG `1g.10gb`) |
| Python | 3.10.12 (`/home/CIN/glp3/replicating-repo-venv/bin/python`) |
| TensorFlow | 2.21.0 |
| Keras | 3.12.4 |
| NumPy / Pandas | 2.2.6 / 2.2.3 |
| Seed | 2019 |
| Modelo | pré-treinado dos autores (`models/new_data_model`) |
| Commit do código | `cdc39d6` (branch `feat/apuana-cluster`) |
| SHA-256 do teste | `e8745596…2499c1` |
| SHA-256 do modelo | `ec2c8074…9520e30a6` |

## 3. Avaliação normal (conjunto completo de teste)

### 3.1 Resumo

Runtime de **2.750,4 s (45,8 min)** para os dois modos, o que equivale a
**~1.306 amostras/s**. O FPR medido coincide com o alvo (0,0100 e 0,1000) por
construção, já que o threshold é o quantil empírico dos benignos.

| Modo | Threshold @1% | Threshold @10% | TPR macro @1% | TPR ponderado @1% | TPR macro @10% | TPR ponderado @10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RePO | 0,019652 | 0,007824 | 0,1384 | 0,0281 | 0,4914 | 0,5012 |
| RePO+ | 0,039149 | 0,006962 | **0,2013** | **0,0711** | **0,6349** | **0,7700** |
| Δ (RePO+ − RePO) | — | — | **+6,28 p.p.** | **+4,30 p.p.** | **+14,35 p.p.** | **+26,88 p.p.** |

A média macro dá o mesmo peso às nove classes maliciosas; a ponderada usa o
número de amostras de cada classe e é dominada por `xss` (60% do teste),
`password` e `injection`.

### 3.2 TPR por classe

| Classe | n | RePO @1% | RePO+ @1% | RePO @10% | RePO+ @10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `backdoor` | 27.145 | 0,3695 | 0,6249 | 0,9945 | **0,9957** |
| `password` | 340.208 | 0,0415 | 0,1344 | 0,6757 | **0,9697** |
| `injection` | 277.696 | 0,0278 | 0,0911 | 0,5890 | **0,9221** |
| `ransomware` | 5.098 | 0,4790 | 0,5947 | 0,7421 | **0,9176** |
| `xss` | 2.149.308 | 0,0198 | 0,0496 | 0,4554 | **0,7171** |
| `scanning` | 36.205 | 0,0770 | 0,1135 | 0,5101 | **0,6865** |
| `mitm` | 517 | 0,2244 | 0,2031 | 0,4371 | **0,5010** |
| `ddos` | 202 | 0,0000 | 0,0000 | 0,0050 | **0,0050** |
| `dos` | 145 | 0,0069 | 0,0000 | 0,0138 | **0,0000** |

### 3.3 Discussão

1. **O RePO+ melhora em quase todas as classes**, e o ganho é maior justamente
   nas classes de TPR intermediário: `password` (+29,4 p.p.), `injection`
   (+33,3 p.p.) e `scanning` (+17,6 p.p.). Em `backdoor`, que já estava em
   0,9945, o teto impede ganho.
2. **A agregação por mínimos por grupo é o que sustenta o TPR alto.** O RePO
   puro, que usa a média das 100 máscaras, fica 14,3 pontos macro abaixo. Isso
   é coerente com a interpretação do artigo: o mínimo por grupo reduz o efeito
   de uma máscara que preservou a região informativa da amostra.
3. **`xss` domina o conjunto** (2,15 M de 3,59 M). Seu TPR de 0,7171 é bom, mas
   é o principal limitador do TPR ponderado (0,7700). Qualquer mudança no TPR
   de `xss` move a média ponderada muito mais que qualquer outra classe.
4. **`ddos` e `dos` não são detectados** nesse threshold (1 acerto em 202 e
   0 em 145). Como essas classes têm 202 e 145 amostras, a estimativa é
   grosseira de qualquer forma, mas o padrão é claro e se repete no RePO.
5. **O salto entre 1% e 10% é grande** (macro 0,2013 → 0,6349). A distribuição
   de escores benignos tem cauda longa; baixar o threshold de 0,0391 para
   0,0070 triplica o TPR. O FPR de 10% é, portanto, o cenário realista para
   comparar com a rodada adversarial, como no protocolo do artigo.

## 4. Avaliação adversarial

### 4.1 Execução

Array SLURM `29541` com nove tarefas (uma por ataque), concorrência 2, cada uma
avaliando até **5.000 amostras aleatórias** (ou todas, quando a classe tem
menos), com `variant=primary`, `sample_mode=random` e
`threshold_policy=calibrated`. O threshold de ataque foi lido de
`normal/repo_plus.json` (`0,0069621`, FPR 10%), com verificação de proveniência
antes de reutilizá-lo.

- Amostras avaliadas: **30.864** (6 ataques com 5.000, `mitm` 517, `ddos` 202,
  `dos` 145).
- Tempo de parede: **31,4 h** (09/11 22:30 → 11/11 05:54, horário local).
- Soma de GPU: **62,5 h**, ou **7,29 s por amostra**.
- Checkpoint a cada 50 amostras; as tarefas reenfileiradas pelo SLURM retomaram
  do checkpoint, e as contagens finais fecham exatamente em `n`.
- Exportação agregada (`29542`) rodou por dependência `afterok` e gerou
  `summary.csv`, `summary.md` e `summary.json` às 05:54.

O custo é dominado pelo laço Python de 540 iterações por amostra; o passo do
otimizador e a projeção estão compilados com `tf.function`, mas as 540
iterações não estão dentro de um `tf.while_loop` (a otimização compilada
existe apenas no caminho de pacotes).

### 4.2 Resultados por ataque

`normal_tpr` é o TPR nas mesmas amostras antes do ataque; `victim_tpr` é o TPR
depois; `queda condicional` é `1 − victim_tpr / normal_tpr` entre as amostras
que o detector já detectava. `fooled` inclui `evaded` e `backup`.

| Ataque | n | TPR normal | TPR vítima | Δ TPR (p.p.) | Queda condicional | fooled | cons_as_ben | Tempo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `backdoor` | 5.000 | 0,9966 | 0,9966 | 0,00 | 0,00% | 0 | 17 | 10,82 h |
| `ransomware` | 5.000 | 0,9172 | 0,9186 | **+0,14** | −0,15% | **3** | 409 | 11,11 h |
| `password` | 5.000 | 0,9680 | 0,9690 | **+0,10** | −0,10% | 0 | 149 | 11,65 h |
| `injection` | 5.000 | 0,9202 | 0,9220 | **+0,18** | −0,20% | 0 | 397 | 11,19 h |
| `xss` | 5.000 | 0,7174 | 0,7180 | **+0,06** | −0,08% | 0 | 1.422 | 8,70 h |
| `scanning` | 5.000 | 0,6972 | 0,6954 | −0,18 | +0,26% | 0 | 1.515 | 8,37 h |
| `mitm` | 517 | 0,5010 | 0,4990 | −0,19 | **+0,39%** | 0 | 259 | 0,58 h |
| `ddos` | 202 | 0,0050 | 0,0000 | −0,50 | +100%* | 0 | 202 | 0,03 h |
| `dos` | 145 | 0,0000 | 0,0000 | 0,00 | n/d | 0 | 145 | 0,02 h |

\* O `ddos` tinha **1 amostra detectada** em 202. A queda condicional de 100%
significa que essa única amostra deixou de ser detectada — não é evidência de
eficácia do ataque. Em `dos` o denominador é zero e a métrica não é definida.

Totais:

| Métrica | Valor |
| --- | ---: |
| Amostras atacadas | 30.864 |
| Evasões (`fooled`/`evaded`) | **3 (0,010%)** |
| `backup` | 0 |
| Já indetectáveis antes do ataque (`cons_as_ben`) | 4.515 (14,6%) |
| Continuaram detectadas (`cons_as_mal`) | 26.346 (85,4%) |
| TPR vítima macro (9 ataques) | 0,6354 |
| TPR vítima macro (7 ataques com TPR normal > 0) | 0,8169 |
| TPR vítima ponderado por amostras | 0,8539 |
| TPR normal ponderado nas mesmas amostras | 0,8535 |
| Δ TPR ponderado (vítima − normal) | +0,04 p.p. |
| Ataques acima do FPR 10% após o ataque | 7 de 9 |

### 4.3 Discussão

1. **Três evasões em 30.864 amostras é indistinguível de ruído.** Todas
   ocorreram em `ransomware`, num conjunto em que 409 amostras (8,2%) já não
   eram detectadas antes do ataque.
2. **A “taxa de sucesso do ataque” engana.** `attack_success_rate = 1 −
   victim_tpr` vale 0,282 no `xss`, mas o TPR normal já era 0,7174; a queda
   condicional é −0,08%. O mesmo vale para todos os ataques: o número aparente
   mede majoritariamente os falsos negativos preexistentes.
3. **Em quatro ataques o TPR “depois” ficou maior que o “antes”** (`ransomware`,
   `password`, `injection`, `xss`), e em `backdoor` empatou. O protocolo reavalia
   cada amostra uma única vez, com outra máscara aleatória do RePO+. Como o
   estimador é estocástico, os desvios observados entre reavaliações (~±0,2 p.p.)
   são da mesma ordem do efeito que se quer medir. O sinal negativo nessas linhas
   não significa que o ataque ajudou o detector — significa que o ataque não
   produziu efeito mensurável.
4. **O ataque otimiza a perda RePO, mas o sucesso é medido pelo RePO+.** A
   perda é a soma das médias dos 100 erros; o agregado RePO+ usa o mínimo de
   cada grupo de 20. Reduzir a média não reduz necessariamente o mínimo: basta
   que algumas máscaras continuem reconstruindo bem a amostra modificada. Essa
   é a explicação mais provável para a assimetria entre “amostra modificada com
   sucesso” e “amostra que evade”. É uma hipótese consistente com os dados, não
   uma demonstração — para testá-la seria preciso medir o erro por máscara
   antes e depois.
5. **`ddos` e `dos` não são casos de sucesso do ataque.** Com TPR normal de
   0,0050 e 0,0000, não há detecção a derrubar. Reportá-los como “100% de
   queda condicional” (`ddos`) ou “taxa de sucesso 1,0” seria um artefato de
   denominador.
6. **O padrão é o oposto do observado no fluxo do CICIDS2017**, onde o
   FTP-Patator derrubou 40,7 p.p. e 82% das detecções. A diferença pede
   cautela: mudam o dataset, as features, a escala dos escores e o próprio
   threshold (aqui calibrado em FPR 10%, lá o threshold publicado × 0,8).

## 5. Comparação entre os três conjuntos

Todos os números abaixo foram produzidos pelo mesmo runner e pelo mesmo
protocolo de máscaras. São datasets diferentes, então a comparação é
**descritiva**, não uma medida de generalização estatística.

### 5.1 Avaliação normal

| Métrica | Fluxo (CICIDS2017) | Pacotes (CICIDS2017) | Novo dataset |
| --- | ---: | ---: | ---: |
| Amostras de teste | 2.298.431 | 2.216.623 | 3.590.733 |
| TPR RePO+ @ FPR 1% | 0,2525 | 0,3472 | 0,2013 (macro) / 0,0711 (pond.) |
| TPR RePO @ FPR 1% | 0,2443 | 0,3185 | 0,1384 (macro) / 0,0281 (pond.) |
| Ganho do RePO+ @ 1% | +0,82 p.p. | +2,87 p.p. | +6,28 p.p. (macro) |
| TPR RePO+ @ FPR 10% | — | — | 0,6349 (macro) / 0,7700 (pond.) |
| Maior TPR por classe @10% | — | — | 0,9957 (`backdoor`) |
| Menor TPR por classe @10% | — | — | 0,0000 (`dos`) |

O ganho absoluto do RePO+ é maior no novo dataset do que nos dois conjuntos do
artigo, mas isso ocorre porque o TPR do RePO puro é muito baixo em `password`,
`injection` e `xss` — classes em que o agregado por mínimos faz mais diferença.
A conclusão qualitativa (RePO+ > RePO) se mantém nos três conjuntos.

### 5.2 Avaliação adversarial

| Métrica | Fluxo (CICIDS2017) | Pacotes (CICIDS2017) | Novo dataset |
| --- | ---: | ---: | ---: |
| Amostras atacadas | 47.003 | 259.890 | 30.864 |
| Ataques | 11 | 11 | 9 |
| TPR vítima ponderado | 0,4663 | 0,5995 | **0,8539** |
| Maior queda absoluta | −40,7 p.p. (FTP-Patator) | −5,39 p.p. (DDoS) | −0,19 p.p. (`mitm`) |
| Maior queda condicional | ~82% (FTP-Patator) | 99,5% (Heartbleed, TPR normal 0,026) | 0,39% (`mitm`) |
| Ataques acima do FPR após ataque | 8 de 11 | 10 de 11 | 7 de 9 |
| Custo por amostra | ~1,16 s (fluxo) | ~1,0 s (pacote) | 7,29 s |

A comparação mais informativa é a **queda condicional**, porque ela desconta os
falsos negativos que já existiam. Nos três conjuntos o RePO+ é atacável em
graus muito diferentes: 82% no pior ataque de fluxo do CICIDS2017, 8,2% no pior
ataque de pacotes com TPR normal relevante (DDoS) e 0,39% no novo dataset.

Duas ressalvas importantes:

1. O TPR vítima ponderado do novo dataset (0,8539) é medido sobre no máximo
   5.000 amostras por ataque. Como `xss` tem 2,15 M amostras no teste completo
   e apenas 5.000 na rodada adversarial, a ponderação é bem diferente da usada
   na avaliação normal (0,7700). Os dois números não devem ser comparados
   diretamente.
2. O custo por amostra do novo dataset é ~6× o do fluxo do artigo, porque o laço
   de 540 iterações roda em Python e a maioria das tarefas recebeu uma fatia
   MIG `1g.10gb` (1/7 da A100). Isso afeta o tempo, não as métricas.

## 6. Reprodutibilidade

- Branch: `feat/apuana-cluster`, commit `cdc39d6`.
- Comandos e decisões: `cluster/README.md` (seção 16) e
  `docs/APUANA_RUNBOOK.md` (seção 7.1).
- Cadeia de jobs: `29540` (normal) → `29541` (array de 9 ataques, concorrência
  2) → `29542` (export), encadeados por `--dependency=afterok`.
- Cada JSON contém `config_hash`, SHA-256 do CSV de teste, SHA-256 do modelo,
  hashes de `train_min`/`train_max`, contagens, tempos, versões do ambiente e
  commit do Git.
- Ambiente exato: `$HOME/replicating-repo-results/env-lock.txt`.
- Artefatos no cluster:

```text
~/replicating-repo-results/new_data/normal/repo.json
~/replicating-repo-results/new_data/normal/repo_plus.json
~/replicating-repo-results/new_data/adversarial/primary/<ataque>.json
~/replicating-repo-results/new_data/scores/*.npz
~/replicating-repo-results/new_data/export/summary.{csv,md,json}
```

- Checkpoints a cada 50 amostras; CSVs extraídos e `.npz` não entram no Git.

## 7. Limitações

1. **Modelo não retreinado.** Toda a avaliação usa o modelo pré-treinado dos
   autores (`models/new_data_model`). O treino reproduzível
   (`train_new_data_model.py`) foi implementado, mas não foi executado nesta
   rodada; não há evidência de que o modelo commitado veio do treino original,
   apenas a validação de sanidade (erro benigno menor que ruído).
2. **Calibração e medição no mesmo conjunto.** Os thresholds são quantis dos
   benignos de `new_test.csv` e o FPR é medido nesses mesmos benignos, como no
   protocolo do artigo. Isso é uma limitação herdada do desenho, não um erro.
3. **Amostragem limitada no adversarial.** Até 5.000 amostras aleatórias por
   ataque. Isso dá boa precisão para as classes grandes, mas a média ponderada
   não representa a composição do teste completo.
4. **Uma única reavaliação por amostra.** Os desvios observados entre
   reavaliações são da ordem de ±0,2 p.p.; os efeitos medidos são menores ou
   iguais a isso. Para afirmar “efeito zero” com confiança seria necessário
   repetir a pontuação com várias sementes de máscara.
5. **Classes minúsculas.** `ddos` (202), `dos` (145) e `mitm` (517) sustentam
   apenas conclusões fracas. Em `ddos` e `dos` a métrica condicional é
   degenerada.
6. **Apenas a variante `primary`.** As sensibilidades `sensitivity_first_n` e
   `sensitivity_paper_0.8` não foram executadas; o export registra 18
   resultados ausentes por isso.
7. **Otimização ausente no ataque.** O laço de 540 iterações por amostra roda
   em Python. A otimização compilada (`tf.while_loop`) existe só no caminho de
   pacotes. O custo é alto (7,29 s/amostra), mas não altera os resultados.
8. **Feature constante com normalização degenerada.** Oito features têm
   `train_max == train_min` e são normalizadas apenas pelo epsilon
   (`31`, `32`, `33`, `48`, `55`, `56`, `57`, `63`). O efeito delas nos escores
   não foi isolado.
9. **Fonte e citação pendentes.** A atribuição oficial ao CIC-ToN-IoT depende
   de confirmação externa.
10. **Ataque otimiza RePO, sucesso medido por RePO+.** É o protocolo do artigo,
    mas deixa em aberto se um atacante que otimizasse diretamente o RePO+
    teria mais sucesso.

## 8. Conclusões

1. **O RePO+ mantém a vantagem sobre o RePO no dataset adicional**: +14,3 p.p.
   de TPR macro no FPR de 10% e +6,3 p.p. no FPR de 1%. A direção do efeito do
   artigo se reproduz fora dos dados originais.
2. **O desempenho é fortemente dependente da classe.** Backdoor, password,
   injection e ransomware ficam acima de 0,91; `xss`, que concentra 60% do
   teste, fica em 0,72; `ddos` e `dos` não são detectados no threshold de 10%.
   A média macro esconde essa dispersão.
3. **O ataque adversarial é praticamente ineficaz neste dataset**: 3 evasões em
   30.864 amostras (0,01%), nenhuma queda de TPR acima de 0,2 p.p. nas classes
   com detecção relevante e queda condicional máxima de 0,39%. Em 7 dos 9
   ataques a detecção continuou acima do FPR.
4. **A métrica de “sucesso do ataque” precisa ser lida com cuidado.** Em todos
   os ataques, `1 − victim_tpr` é dominado pelos 14,6% de amostras que já não
   eram detectadas. A queda condicional é a métrica adequada, e ela mostra que
   o ataque não converte modificação em evasão.
5. **O contraste com o CICIDS2017 é o achado mais relevante para o relatório
   final.** No fluxo do artigo, um ataque derrubou 82% das detecções; aqui, no
   máximo 0,39%. Isso mostra que a robustez do RePO+ medida no artigo **não se
   transfere automaticamente** para outros conjuntos e precisa ser reavaliada
   caso a caso.
6. **O custo do ataque é o gargalo operacional**: 62,5 h de GPU para 30.864
   amostras, contra 45,8 min para pontuar 3,59 M amostras no modo normal. Uma
   rodada de sensibilidades ou um ataque adaptativo ao RePO+ exige orçamento
   próprio.
7. **Próximos passos naturais**: (i) rodar as duas sensibilidades de amostragem
   e de threshold; (ii) retreinar o autoencoder no `new_train.csv` e repetir a
   avaliação; (iii) medir o erro por máscara antes e depois do ataque para
   testar a hipótese da seção 4.3; (iv) repetir a pontuação com múltiplas
   sementes para separar ruído de efeito.
