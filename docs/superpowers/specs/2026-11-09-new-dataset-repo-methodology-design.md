# Avaliação do novo dataset com a metodologia RePO/RePO+

**Data:** 2026-11-09  
**Branch de implementação:** `feat/new-dataset-eval` (derivada de `origin/new-dataset`)  
**Status:** design proposto; implementação bloqueada até aprovação

## 1. Intenção

Permitir que o novo dataset seja avaliado com a mesma metodologia do artigo de
referência — treino do autoencoder apenas com benignos, avaliação normal com
FPR alvo de 1%, avaliação adversarial com FPR alvo de 10% e ataque RePO+ — de
forma reproduzível e auditável, sem depender do cluster Apuana nesta etapa.

O resultado esperado é um conjunto de resultados normais e adversariais por
classe de ataque, com thresholds calculados a partir do próprio modelo,
métricas definidas sem ambiguidade e artefatos suficientes para alimentar o
relatório.

## 2. Contexto

O branch `origin/new-dataset` contém um protótipo:

- `code/train_new_data_model.py` — treina o autoencoder no novo dataset;
- `code/new_data_RePO_Normal_Test.ipynb` — avaliação normal;
- `code/new_data_RePO_Adversarial.ipynb` — avaliação adversarial;
- `data/new_data/preprocess_new_data.py` — script de pré-processamento;
- `data/new_data/New-dataset-split-clean.zip` — dados via Git LFS;
- `models/new_data_model/` — modelo treinado;
- `data/new_data/x_train_meta/train_{min,max}.npy` — normalização de 77 features.

O ZIP **não contém o dataset bruto**: ele contém os CSVs já processados
`new_train.csv` (~896 MB descomprimido) e `new_test.csv` (~2,1 GB
descomprimido), ambos com 85 colunas no layout CICIDS:

- coluna 0: `column00`;
- colunas 1–7: identificadores e metadados (`Flow ID`, IPs, portas, protocolo,
  timestamp);
- colunas 8–83: 76 features de fluxo;
- coluna 84: `Label`.

As 77 features usadas pelo modelo são `Dst Port` (índice 5) mais as colunas
8–83. O `Label` fica de fora.

Os rótulos observados são todos minúsculos:

`BENIGN`, `xss`, `password`, `injection`, `scanning`, `backdoor`,
`ransomware`, `mitm`, `ddos`, `dos`.

O modelo existente foi treinado com 77 features e o `new_train.csv` contém
somente `BENIGN`. Ainda assim, o pipeline commitado não é executável nem
reproduzível como está.

## 3. Objetivos

- **G1 — Dados:** extrair, validar e documentar os CSVs do novo dataset.
- **G2 — Avaliação normal:** calcular RePO e RePO+ no conjunto de teste,
  thresholds de FPR 1% e 10%, TPR por classe, FPR e contagens.
- **G3 — Avaliação adversarial:** executar RePO+ por classe de ataque, com
  threshold calibrado no modelo novo, amostragem controlada, checkpoint e
  resultados por ataque.
- **G4 — Reprodutibilidade:** seed fixa, hashes de dados/modelo/metadados,
  manifesto de ambiente e resultados em schema estável.
- **G5 — Correções metodológicas:** eliminar os erros conhecidos dos notebooks
  atuais (normalização ausente, threshold hardcoded, label com caixa errada,
  resultado no arquivo de fluxo, métrica duplicada, viés das primeiras N
  amostras).
- **G6 — Entrega:** testes automatizados, documentação do dataset e
  instruções de execução.

## 4. Não objetivos

- **Não** criar ou alterar o runner/SLURM do Apuana (`cluster/`).
- **Não** avaliar o dataset de pacotes.
- **Não** propor nova arquitetura de modelo nem fazer busca de
  hiperparâmetros.
- **Não** reproduzir o ZIP a partir do dataset bruto nesta entrega; o ZIP é a
  fonte canônica dos CSVs processados.
- **Não** alterar os resultados já existentes de fluxo/pacotes do artigo.
- **Não** substituir o relatório; apenas gerar os artefatos que ele consumirá.

## 5. Decisões tomadas

O parceiro optou por não responder perguntas antes do design. As decisões
abaixo são escolhas do design e devem ser corrigidas antes da implementação se
não representarem o objetivo real.

### D1 — Ataques avaliados

Serão avaliadas as **9 classes maliciosas do novo dataset**: `xss`,
`password`, `injection`, `scanning`, `backdoor`, `ransomware`, `mitm`, `ddos`
e `dos`.

A metodologia do artigo é definida pelo protocolo (treino benigno, RePO/RePO+,
FPR 1%/10%, TPR por ataque), não pelos 11 nomes de ataque do CICIDS2017. Não
faz sentido recriar FTP-Patator, Heartbleed etc. num dataset que não os contém.

### D2 — Fonte de dados

O ZIP `New-dataset-split-clean.zip` é a fonte canônica para execução. O
`preprocess_new_data.py` não será necessário para rodar; ele será corrigido ou
marcado como legado para não sugerir que reproduz o ZIP quando não reproduz.

### D3 — Amostragem adversarial

O artigo usa as primeiras 5.000 amostras de cada ataque. Isso introduz viés
temporal, já documentado em `docs/REPLICATION_ANALYSIS.md`.

Decisão: a execução principal usará **até 5.000 amostras aleatórias por
ataque**, com seed fixa. Quando o ataque tiver menos de 5.000 amostras, usa
todas. A execução com as primeiras 5.000 será mantida como análise de
sensibilidade opcional, não como resultado principal.

### D4 — Thresholds

Nenhum threshold será copiado do modelo de fluxo antigo. Todos serão
calculados sobre os escores benignos do modelo novo:

- normal: threshold no FPR de 1%;
- adversarial: threshold no FPR de 10%;
- o fator 0,8 do notebook de fluxo do artigo será tratado como sensibilidade,
  não como regra principal, porque ele foi ajustado empiricamente para outro
  modelo.

Será medido e reportado o FPR real obtido em cada threshold.

### D5 — Implementação

A lógica será extraída para um pacote Python em `code/new_data_eval/`, com CLI.
Os notebooks passarão a ser finos, chamando as mesmas funções para inspeção e
smoke tests; eles não serão a implementação de referência.

Isso evita duplicar a avaliação em dois notebooks e facilita uma futura
integração com o Apuana sem reescrever a metodologia.

### D6 — Modelo

O modelo existente será reutilizado por padrão. Antes de qualquer avaliação,
será validado contra os CSVs e os metadados. O script de treino será corrigido
para ser reproduzível, mas o retreinamento não é obrigatório nesta entrega.

### D7 — Escopo de cluster

Nada em `cluster/` será alterado. A execução será local/Colab, com suporte a
GPU. A integração com o Apuana fica para uma etapa posterior.

### D8 — Loss do ataque versus métrica de sucesso

O ataque preservará o comportamento de referência do notebook/runner:

- a **loss otimizada** é a soma dos 100 erros médios de reconstrução
  (objetivo no estilo RePO);
- a **métrica de sucesso e o TPR da vítima** usam o agregado RePO+
  (mínimo por grupo de 20, soma dos 5 mínimos).

Ou seja, “ataque RePO+” significa que a avaliação é RePO+, não que a loss
otimizada seja o agregado RePO+. Isso mantém comparabilidade com o artigo e com
o runner do Apuana. Otimizar diretamente o RePO+ fica fora do escopo.

### D9 — Máscaras e reprodutibilidade

Para evitar que o resultado dependa do tamanho do lote:

- cada amostra recebe 100 máscaras independentes, geradas com RNG
  *stateless* a partir de `(seed, sample_index, phase, repeat)`;
- `phase` distingue escore original, checagem durante o ataque e escore da
  vítima;
- o ataque usa uma nova máscara a cada iteração, também derivada de
  `(seed, sample_index, iteration)`;
- `--batch-size` não pode alterar os escores.

O comportamento principal usa uma pontuação por fase (como o artigo), mas a
CLI terá `--score-repeats` (padrão 1) para uma análise de ruído com múltiplas
pontuações independentes. `--paired-scoring` (padrão desligado) reutiliza a
mesma semente nas fases antes/depois, como sensibilidade.

### D10 — Calibração dos thresholds

Os thresholds são calibrados nos benignos de `new_test.csv`, que ficaram fora
do treino (o split separou 30% dos benignos para teste). A calibração e a
medição do FPR usam o mesmo conjunto, como no protocolo do artigo; isso é uma
limitação registrada, não um erro acidental.

O resultado normal grava os thresholds de FPR 1% e 10%. O adversarial **lê o
threshold de FPR 10% do JSON normal** por padrão, para garantir que as duas
etapas usem exatamente o mesmo valor. Se o JSON não existir, o adversarial
recalcula e registra aviso.

### D11 — Validação e retreino do modelo

A validação mínima do modelo existente é:

- carregar sem erro;
- aceitar entrada `(None, 77)`;
- produzir saídas finitas;
- ter erro de reconstrução em amostras benignas de treino menor que o erro em
  ruído aleatório (sanidade, não prova de qualidade).

Hashes de dados, metadados e modelo são registrados. Como não existe hash de
referência do treino original, a validação não prova proveniência. Se a
sanidade falhar, a avaliação para e o retreino passa a ser uma etapa explícita
e separada. Retreinar usa seed 2019, 5 epochs e batch 256 (os valores do script
atual); se os CSVs de treino não mudarem, os `.npy` de normalização commitados
continuam sendo a fonte.

### D12 — Médias e resultados de sensibilidade

As médias normais e adversariais são calculadas **somente sobre as 9 classes
maliciosas**, nunca incluindo `BENIGN`:

- `macro`: média simples das 9 TPRs;
- `weighted`: média ponderada pelo número de amostras de cada classe.

Classes com `n < 100` entram no resultado principal com `low_n: true` e também
aparecem numa média macro de sensibilidade sem classes pequenas. O `n` sempre
será exibido.

Execuções de sensibilidade terão diretório próprio, para não sobrescrever o
resultado principal:

```text
results/new_data/adversarial/primary/<ataque>.json
results/new_data/adversarial/sensitivity_first_n/<ataque>.json
results/new_data/adversarial/sensitivity_paper_0.8/<ataque>.json
```

Cada JSON terá o campo `variant`. O agregador ignora variantes que não sejam
`primary`, a menos que seja solicitado explicitamente.

### D13 — Identidade documental do dataset

O script legado e os headers indicam **CIC-ToN-IoT**. A origem oficial,
licença, citação e justificativa da escolha ainda precisam ser confirmadas com
quem produziu o branch. Isso não bloqueia a implementação, mas bloqueia a
redação final do relatório. O `docs/NEW_DATASET.md` terá uma seção “Fonte e
citação” marcada como pendente até essa confirmação.

## 6. Protocolo metodológico

### 6.1 Dados e features

- `new_train.csv`: usado apenas para treino/reprodução; deve conter somente
  `BENIGN`.
- `new_test.csv`: usado na avaliação; contém 30% dos benignos e todos os
  maliciosos.
- Features: `Dst Port` + colunas 8–83 = 77 features.
- Normalização: `(x - train_min) / (train_max - train_min + 1e-6)`, usando os
  `.npy` commitados.
- Linhas com `inf`/`nan` nas features são descartadas e a quantidade é
  registrada.
- Rótulos são normalizados apenas para `BENIGN`; os nomes dos ataques são
  preservados e comparados sem diferenciar maiúsculas/minúsculas, após
  `strip`.

### 6.2 RePO e RePO+

Para cada amostra, são geradas 100 máscaras binárias independentes, com cada
feature mascarada com probabilidade 0,75. O erro de reconstrução é a média
quadrática por amostra e por máscara.

- **RePO:** média dos 100 erros.
- **RePO+:** os 100 erros são divididos em 5 grupos de 20; toma-se o mínimo de
  cada grupo e soma-se os 5 mínimos.

As máscaras são geradas por amostra, com RNG stateless e semente derivada de
`(seed, sample_index, phase, repeat)`. Assim, o escore de uma amostra não
depende do lote em que ela foi processada, e a mesma seed reproduz o mesmo
resultado. O ataque usa uma máscara nova por iteração, derivada de
`(seed, sample_index, iteration)`.

A loss otimizada pelo ataque é a soma dos 100 erros médios (RePO); a checagem
de sucesso e o TPR da vítima usam o agregado RePO+ (D8).

### 6.3 Avaliação normal

1. Carregar `new_test.csv` em chunks, selecionar as 77 features, normalizar e
   pontuar todas as amostras com RePO e RePO+.
2. Calcular o threshold de FPR 1% sobre os escores benignos.
3. Calcular o threshold de FPR 10% sobre os benignos (para uso adversarial).
4. Reportar, **para cada um dos dois thresholds**:
   - valor do threshold;
   - FPR obtido;
   - TPR por classe;
   - contagem por classe.
5. Reportar as médias `macro` e `weighted` apenas sobre as 9 classes
   maliciosas, com `n` por classe e a marcação `low_n` para `n < 100`.
6. Salvar escores e JSON.

A calibração usa os benignos de `new_test.csv`, que não foram vistos no treino.
O FPR reportado é medido no mesmo conjunto usado para calibrar, como no
protocolo do artigo; o design registra essa limitação. Uma análise de
sensibilidade opcional pode dividir os benignos de teste em metade de
calibração e metade de avaliação, mas isso não é o resultado principal.

O conjunto de teste completo será usado; não haverá truncamento para 2.298.431
amostras nem para as primeiras N.

### 6.4 Avaliação adversarial

Para cada uma das 9 classes maliciosas:

1. Selecionar até 5.000 amostras aleatórias da classe, com seed fixa; se houver
   menos, usar todas.
2. Calcular o escore original com RePO+.
3. Construir o ataque RePO+ com:
   - `real_threshold` = threshold de FPR 10% lido do resultado normal;
   - `attack_threshold` = o mesmo threshold calibrado de FPR 10% (decisão D4);
   - com `--attack-threshold-policy paper_0.8`,
     `attack_threshold = 0.8 * real_threshold` apenas como sensibilidade.
4. Otimizar cada amostra com a mesma cadeia de máscaras e otimizadores do
   notebook/runner de fluxo:
   - `mask_l1`, `mask_l2`, `mask_l3`;
   - Adam com learning rates 0,001; 0,01; 0,1; 1,0;
   - 30, 40, 50 e 60 iterações por máscara;
   - mesmas regras de features duplicadas, mínimos, máximos, flags,
     features agregadas e recálculo de features dependentes;
   - loss = soma dos 100 erros médios (RePO), como em D8.
5. Reavaliar o exemplo adversarial com RePO+.
6. Salvar checkpoint a cada 50 amostras e permitir retomada; o checkpoint
   guarda o índice, os escores, as contagens e o estado dos otimizadores.
7. Reportar, por ataque:
   - `normal_tpr_at_attack_threshold`;
   - `victim_tpr`;
   - `attack_success_rate = 1 - victim_tpr`;
   - `conditional_drop = 1 - victim_tpr / normal_tpr_at_attack_threshold`,
     quando o denominador for maior que zero;
   - `fooled_rate`, separando exemplos abaixo do threshold de ataque,
     exemplos de backup (abaixo do threshold real, mas acima do de ataque) e
     “no change”;
   - contagens (`n`, `fooled`, `backup`, `cons_as_mal`, `cons_as_ben`);
   - threshold e FPR reais;
   - runtime.

`attack_success_rate` não será apresentado como “taxa de sucesso do ataque”
isoladamente, porque inclui falsos negativos que já existiam antes do ataque.
A métrica principal de evasão é a queda condicional.

### 6.5 Avaliação do ataque

O ataque é considerado “fooled” quando o otimizador retorna um exemplo
modificado. Dentro desse grupo, distinguimos:

- `evaded`: exemplo abaixo do threshold de ataque;
- `backup`: exemplo abaixo do threshold real, mas não do threshold de ataque;
- `no_change`: amostra já estava abaixo do threshold antes do ataque;
- `not_fooled`: o otimizador esgotou as tentativas sem retornar exemplo.

`fooled_rate` no schema é a fração de amostras com exemplo modificado
(`evaded + backup`), preservando a semântica do runner de referência. O
`victim_tpr` já captura o efeito final: apenas exemplos abaixo do threshold de
ataque deixam de ser detectados. O relatório deve mostrar as duas coisas, sem
tratar `1 - victim_tpr` como taxa de sucesso do ataque.

Quando `--attack-threshold-policy paper_0.8` for usado, os exemplos de backup
são mais prováveis; eles continuam contados como `backup`, não como `evaded`.

### 6.6 Sensibilidade

Serão relatados como análises de sensibilidade, não como resultado principal:

- amostragem “primeiras 5.000” versus aleatória;
- threshold com fator 0,8 versus threshold calibrado de FPR 10%;
- ataques com menos de 100 amostras (reportar n e evitar conclusões fortes);
- múltiplas pontuações independentes (`--score-repeats > 1`) e pontuação
  pareada (`--paired-scoring`).

Cada sensibilidade grava em seu próprio diretório e carrega `variant` no JSON,
seguindo D12. O resultado principal é sempre `variant="primary"`.

## 7. Arquitetura proposta

```text
extract_new_data.sh                  # LFS + unzip + validação
code/new_data_eval/
  __init__.py
  __main__.py                        # PYTHONPATH=code python -m new_data_eval
  data.py                            # CSV, features, labels, metadados, hashes
  scoring.py                         # máscaras, RePO, RePO+, thresholds
  attack.py                          # FlowAttack portado e corrigido
  metrics.py                         # TPR, FPR, médias, queda condicional
  results.py                         # schema, JSON atômico, manifesto
  cli.py                             # validate, normal, adversarial, smoke
  tests/
    test_data.py
    test_scoring.py
    test_metrics.py
    test_results.py
    test_attack_smoke.py
docs/NEW_DATASET.md                  # origem, classes, split, protocolo, comandos
results/new_data/
  normal/repo.json
  normal/repo_plus.json
  adversarial/primary/<ataque>.json
  adversarial/sensitivity_first_n/<ataque>.json
  adversarial/sensitivity_paper_0.8/<ataque>.json
  manifest.json
  scores/*.npz                       # ignorados no Git
```

### 7.1 Fluxo de dados

```text
ZIP LFS
  -> extract_new_data.sh
  -> data/new_data/new_train.csv + new_test.csv
  -> validação de header, features, labels, metadados e modelo
  -> CLI normal    -> results/new_data/normal/*.json
  -> CLI adversarial -> results/new_data/adversarial/<variant>/*.json
```

### 7.2 CLI

O pacote fica em `code/new_data_eval/`, mas é importado como `new_data_eval`
com `PYTHONPATH=code`. Não criar `code/__init__.py`: isso sombrearia o módulo
`code` da biblioteca padrão.

```bash
bash extract_new_data.sh
PYTHONPATH=code python -m new_data_eval validate
PYTHONPATH=code python -m new_data_eval normal --modes repo,repo_plus
PYTHONPATH=code python -m new_data_eval adversarial --attack ddos --max-samples 100
PYTHONPATH=code python -m new_data_eval adversarial --attack all
PYTHONPATH=code python -m new_data_eval smoke
```

Opções previstas:

- `--data-dir`, `--model-dir`, `--meta-dir`, `--results-dir`;
- `--seed` (padrão 2019);
- `--max-samples` (padrão 5000 no adversarial);
- `--sample-mode random|first` (padrão `random`);
- `--batch-size`;
- `--n-masks` (padrão 100; usado apenas em testes/smoke);
- `--attack-threshold-policy calibrated|paper_0.8` (padrão `calibrated`);
- `--score-repeats` (padrão 1);
- `--paired-scoring` (padrão desligado);
- `--variant` (padrão `primary`);
- `--checkpoint-every` (padrão 50);
- `--force`;
- `--dry-run`.

## 8. Correções necessárias

| Problema atual | Correção | Por quê |
|---|---|---|
| ZIP é ponteiro LFS e não há extração | `extract_new_data.sh` com `git lfs pull`, unzip e validação | Sem os CSVs, nenhum notebook roda; no Apuana isso será ainda mais crítico |
| `preprocess_new_data.py` usa `row_offset` antes de definir | Inicializar `row_offset = 0` | O script falha com `NameError` |
| Caminho fixo `/content/replicating_RePO` | Usar caminho relativo ao repositório ou argumento | Não funciona fora do Colab |
| Extração verifica `data/` em vez do CSV | Verificar o arquivo de dados esperado | A condição atual nunca dispara |
| ZIP de origem não existe no branch | Documentar que o ZIP oficial é `New-dataset-split-clean.zip`; remover caminho antigo | O script procura um arquivo inexistente |
| Dependências `tqdm`/`scikit-learn` ausentes | Adicionar ao `requirements.txt` se o script for mantido | O script não instala o que usa |
| Notebook normal não salva nada | Persistir JSON/CSV com thresholds, FPR, TPR e contagens | Sem artefato não há evidência reproduzível |
| `real_thr` hardcoded de 0,016235… | Calcular threshold de FPR 10% no modelo novo | O valor pertence ao modelo de fluxo antigo |
| `attack_type = 'DDoS'` | Usar `ddos` ou derivar labels do CSV; comparar sem caixa | O filtro atual retorna vazio |
| Resultado adversarial salvo em `flow_based_adversarial_results.json` | Salvar em `results/new_data/adversarial/<variant>/` com `model_type="new_data_flow"` | Contamina os resultados de fluxo existentes |
| `model_type="flow"` no resultado | `model_type="new_data_flow"` | O dataset é outro; agregação e relatório precisam distinguir |
| Notebook adversarial roda só um ataque | CLI com `--attack all` e um JSON por ataque | A metodologia exige todos os ataques |
| Sem checkpoint/retomada | Checkpoint a cada 50 amostras | Evita perder horas de GPU em interrupções |
| Sem seed | `tf.keras.utils.set_random_seed(2019)` e seeds por amostra | Resultados não são reproduzíveis |
| Lista de labels hardcoded e frágil | Derivar do CSV, validar conjunto esperado, comparar case-insensitive | Evita classes vazias silenciosas |
| Índices de features fixos | Validar `num_input`, comprimento das máscaras e índices; falhar se não bater | Ataque inválido se a ordem das features mudar |
| Sempre usa as primeiras N amostras | Amostragem aleatória com seed; primeiras N como sensibilidade | Viés temporal documentado no CICIDS2017 |
| Sem documentação do novo dataset | `docs/NEW_DATASET.md` com origem, split, contagens e protocolo | Requisito do relatório e da checklist |
| Modelo sem proveniência | Manifesto com seed, versões, hashes, epochs e loss | Difícil defender o resultado sem saber o que foi treinado |
| Loss do ataque confundida com métrica de sucesso | Loss RePO, sucesso/TPR em RePO+ (D8) | Preserva o protocolo e evita chamar de “RePO+” um ataque otimizado para outra métrica |
| Máscaras dependentes do lote | RNG stateless por amostra/fase/iteração (D9) | `--batch-size` não pode mudar o resultado; checkpoint deve reproduzir |
| Threshold adversarial copiado do normal sem vínculo | Adversarial lê o FPR 10% do JSON normal (D10) | Garante que as duas etapas usem o mesmo threshold |
| Validação do modelo baseada só em shape | Sanidade de reconstrução + hashes (D11) | Shape sozinho não detecta modelo aleatório ou incompatível |
| Médias incluindo BENIGN | Médias só nas 9 classes maliciosas, macro e weighted (D12) | BENIGN e `xss` dominariam o número e esconderiam os ataques |
| Sensibilidades sobrescrevem o resultado principal | Diretórios e `variant` separados (D12) | Evita que uma execução `first_n` ou `paper_0.8` seja confundida com o resultado principal |
| Origem/citação do dataset não documentada | Seção “Fonte e citação” em `docs/NEW_DATASET.md`, marcada como pendente (D13) | O relatório exige origem e justificativa; não inventar atribuição |

## 9. Schema de resultados

### 9.1 Resultado normal

```json
{
  "schema_version": "new_data_v1",
  "status": "completed",
  "experiment": "normal",
  "model_type": "new_data_flow",
  "score_mode": "repo_plus",
  "dataset": {
    "name": "new_data",
    "test_csv": "data/new_data/new_test.csv",
    "test_csv_sha256": "...",
    "n_samples": 0,
    "n_dropped_invalid": 0
  },
  "thresholds": {
    "fpr_0.01": 0.0,
    "fpr_0.10": 0.0
  },
  "fpr": {
    "at_fpr_0.01_threshold": 0.01,
    "at_fpr_0.10_threshold": 0.10
  },
  "class_rates": {
    "at_fpr_0.01": {
      "BENIGN": {"n": 0, "tpr": 0.0, "fpr": 0.0},
      "ddos": {"n": 0, "tpr": 0.0, "low_n": false}
    },
    "at_fpr_0.10": {
      "BENIGN": {"n": 0, "tpr": 0.0, "fpr": 0.0},
      "ddos": {"n": 0, "tpr": 0.0, "low_n": false}
    }
  },
  "average_tpr": {
    "at_fpr_0.01": {"macro": 0.0, "weighted": 0.0, "macro_excluding_low_n": 0.0},
    "at_fpr_0.10": {"macro": 0.0, "weighted": 0.0, "macro_excluding_low_n": 0.0}
  },
  "scoring": {
    "n_masks": 100,
    "score_repeats": 1,
    "paired_scoring": false
  },
  "variant": "primary",
  "runtime_seconds": 0.0,
  "config": {},
  "environment": {},
  "config_hash": "...",
  "git_commit": "...",
  "started_at": "...",
  "finished_at": "..."
}
```

### 9.2 Resultado adversarial

```json
{
  "schema_version": "new_data_v1",
  "status": "completed",
  "experiment": "adversarial",
  "model_type": "new_data_flow",
  "score_mode": "repo_plus",
  "attack_type": "ddos",
  "variant": "primary",
  "dataset": {
    "n_available": 0,
    "n_samples": 0,
    "sample_mode": "random",
    "seed": 2019
  },
  "thresholds": {
    "fpr_0.10": 0.0,
    "attack": 0.0,
    "policy": "calibrated"
  },
  "scoring": {
    "score_mode": "repo_plus",
    "n_masks": 100,
    "score_repeats": 1,
    "paired_scoring": false
  },
  "fpr_at_attack_threshold": 0.0,
  "normal_tpr_at_attack_threshold": 0.0,
  "victim_tpr": 0.0,
  "attack_success_rate": 0.0,
  "conditional_drop": 0.0,
  "fooled_rate": 0.0,
  "evaded_rate": 0.0,
  "backup_rate": 0.0,
  "counts": {
    "n": 0,
    "fooled": 0,
    "evaded": 0,
    "backup": 0,
    "cons_as_mal": 0,
    "cons_as_ben": 0
  },
  "runtime_seconds": 0.0,
  "config": {},
  "environment": {},
  "config_hash": "...",
  "git_commit": "...",
  "started_at": "...",
  "finished_at": "..."
}
```

Regras:

- escrita atômica e um arquivo por execução; não usar append;
- `config_hash` cobre seed, thresholds, amostragem, modelo, dados e versão do
  código;
- escores ficam em `.npz` para auditoria; JSON é o artefato versionado;
- resultados existentes de fluxo/pacotes não podem ser alterados.

## 10. Critérios de aceitação

1. Em um clone limpo, `bash extract_new_data.sh` produz
   `data/new_data/new_train.csv` e `new_test.csv` e a validação passa.
2. `PYTHONPATH=code python -m new_data_eval validate` confirma:
   - 85 colunas e nomes esperados;
   - 77 features selecionadas;
   - `new_train.csv` só com `BENIGN`;
   - todas as 9 classes no teste;
   - metadados com shape `(77,)`, finitos e `max >= min`; features constantes
  (`max == min`) são aceitas e reportadas, porque os `.npy` commitados contêm
  oito delas;
   - modelo carrega e aceita `(None, 77)`.
3. `normal` gera dois JSONs (`repo` e `repo_plus`) com thresholds de 1% e 10%,
   FPR, TPR por classe **em cada threshold**, contagens, médias macro/weighted
   das 9 classes e runtime.
4. `adversarial --attack ddos --max-samples 100` conclui e grava
   `results/new_data/adversarial/primary/ddos.json`, sem tocar em resultados de
   fluxo. O JSON deve registrar `model_type="new_data_flow"`,
   `variant="primary"` e os thresholds lidos do resultado normal.
5. `adversarial --attack all` percorre as 9 classes, lida com classes de
   poucas amostras, separa `evaded`/`backup`/`no_change` e retoma de
   checkpoint.
6. Duas execuções de smoke com a mesma seed produzem as mesmas métricas dentro
   da tolerância definida, e `--batch-size` diferente não altera os escores.
7. Uma execução de sensibilidade (`first_n` ou `paper_0.8`) grava em diretório
   próprio e não sobrescreve `primary`.
8. Testes automatizados passam.
9. `docs/NEW_DATASET.md`, `README.md` e a seção 5 de
   `docs/PROJECT_TASKS.md` foram atualizados.
10. Nenhuma alteração em `cluster/`.

## 11. Plano de testes

### Unitários

- seleção de features e labels a partir de um CSV sintético com 85 colunas;
- normalização com `train_min`/`train_max`;
- `threshold_at_fpr` (método `ceil`) com distribuições conhecidas;
- agregação RePO e RePO+ com matriz de erros controlada;
- `conditional_drop` com denominador zero;
- normalização de labels e comparação case-insensitive;
- amostragem aleatória reproduzível;
- schema e escrita atômica de JSON.

### Integração

- `validate` com CSV correto e com header errado;
- smoke com 20 benignos e 5 amostras de um ataque, `--n-masks 5`,
  `--max-samples 5`;
- adversarial com classe inexistente deve falhar com mensagem útil;
- checkpoint: interromper no meio, retomar e comparar resultado final.

### Regressão

- garantir que `code/experiment_results.py` continue compatível com os
  notebooks de fluxo/pacotes existentes;
- garantir que `results/flow_based_adversarial_results.json` e
  `results/packet_based_adversarial_results.json` não sejam modificados.

## 12. Documentação

Criar `docs/NEW_DATASET.md` com:

- origem e justificativa do dataset;
- o que o ZIP contém (CSVs processados, não brutos);
- split: treino benigno, teste com 30% dos benignos + todos os maliciosos;
- contagens por classe (geradas pelo validador);
- features usadas e normalização;
- protocolo normal/adversarial;
- comandos de execução;
- limitações (classe desbalanceada, amostras pequenas, sem cluster).

Atualizar:

- `README.md` com extração e comandos;
- `docs/PROJECT_TASKS.md`, seção 5, após os resultados;
- `report.tex`, subseção “Novo conjunto de dados escolhido” e resultados,
  quando os números existirem.

## 13. Riscos e limites

1. **Premissa dos 9 ataques:** se o objetivo for mapear os 11 ataques do
   artigo para o novo dataset, o escopo muda e não há correspondência direta.
2. **Origem/citação do dataset:** o nome CIC-ToN-IoT vem do script legado, mas
   fonte oficial e licença ainda não foram confirmadas; isso bloqueia a
   redação final, não a implementação.
3. **Dataset desbalanceado:** `xss` domina; classes como `mitm` podem ter
   pouquíssimas amostras. Reportar `n` e evitar médias macro sem contexto.
4. **Custo computacional:** o teste tem ~2,1 GB e o adversarial pode levar
   horas. Checkpoint e amostragem são obrigatórios.
5. **Modelo existente pode não corresponder aos CSVs:** a validação por shape,
   sanidade e hashes detecta casos grosseiros; se falhar, retreinar é
   necessário.
6. **Sem Apuana:** a execução local/Colab pode ser lenta; a prioridade é
   corrigir a metodologia, não otimizar o cluster.
7. **Amostragem aleatória ainda tem variância:** reportar seed e, se possível,
   uma segunda amostragem para ataques pequenos.
8. **Threshold calibrado e avaliado no mesmo conjunto benigno:** é o protocolo
   do artigo; a limitação deve constar no relatório e pode ser tratada com a
   análise de sensibilidade de metades separadas.

## 14. Ordem de implementação

1. `extract_new_data.sh` + validação + `docs/NEW_DATASET.md` inicial.
2. Pacote `code/new_data_eval/`: `data.py`, `scoring.py`, `metrics.py`,
   `results.py` e testes unitários.
3. CLI `validate` e `normal`; gerar os primeiros resultados normais.
4. Portar `FlowAttack` para `attack.py` e rodar smoke adversarial.
5. CLI `adversarial` com checkpoint, variantes e `--attack all`.
6. Corrigir `train_new_data_model.py` para manifesto/seed (retreino opcional;
   só executar se a validação do modelo falhar).
7. Atualizar notebooks para chamar o pacote; remover lógica duplicada.
8. Rodar smoke, depois a avaliação normal completa e os 9 ataques.
9. Rodar as sensibilidades `first_n` e `paper_0.8` se houver tempo/orçamento.
10. Atualizar docs/checklist/relatório com os resultados.
11. (Fora de escopo) integração com o Apuana.

## 15. Fora de escopo

- `cluster/`, SLURM, Apuana;
- dataset de pacotes;
- novas arquiteturas ou hiperparâmetros;
- reprodução do dataset bruto;
- alteração dos resultados de fluxo/pacotes do artigo;
- escrita final do relatório.
