# Integração do novo dataset ao Apuana

**Data:** 2026-11-09  
**Branch de integração:** `feat/apuana-cluster`  
**Status:** design proposto; implementação bloqueada até aprovação

## 1. Intenção

Permitir que a avaliação RePO/RePO+ do novo dataset (`data/new_data/`) rode no
cluster Apuana com o mesmo tooling operacional já usado para fluxo e pacotes:
setup no nó de login, jobs SLURM com GPU, checkpoint/retomada, agregação e
coleta de resultados.

O objetivo não é reimplementar a avaliação. O pacote `code/new_data_eval/` já
contém validação, scoring, ataque, métricas e CLI testados. A entrega é fazer
esse pacote conviver com `cluster/`, extrair os dados no `$SCRATCH`, expor
comandos de submissão e produzir resultados no mesmo padrão auditável dos
experimentos existentes.

## 2. Contexto

A branch `feat/new-dataset-eval` contém:

- `code/new_data_eval/` — pacote e CLI (`validate`, `normal`, `adversarial`,
  `smoke`);
- `code/train_new_data_model.py` — treino reproduzível do autoencoder;
- `extract_new_data.sh` — extração do ZIP LFS;
- `data/new_data/New-dataset-split-clean.zip` — ZIP via Git LFS;
- `data/new_data/x_train_meta/train_{min,max}.npy` — normalização de 77
  features;
- `models/new_data_model/` — SavedModel pré-treinado;
- notebooks wrappers, testes, docs e resultados de smoke locais.

A branch `feat/apuana-cluster` contém:

- `cluster/setup_env.sh` — venv, dados de fluxo/pacotes, symlinks, preflight;
- `cluster/submit.sh` — submissão de treino, normal, adversarial e export;
- `cluster/slurm/job.sh` — entrypoint dos jobs;
- `cluster/common.sh` — paths, modelos, validações e ambiente;
- `cluster/aggregate_results.py` — sumarização de fluxo/pacotes;
- `cluster/README.md` e `docs/APUANA_RUNBOOK.md`.

As duas branches divergem em 30 commits (Apuana) e 42 commits (novo dataset),
com um único conflito previsto: `docs/PROJECT_TASKS.md`.

## 3. Objetivos

- **G1 — Merge:** trazer `feat/new-dataset-eval` para `feat/apuana-cluster`
  preservando histórico, testes e docs, sem alterar resultados de fluxo ou
  pacotes.
- **G2 — Dados:** extrair `new_train.csv` e `new_test.csv` para
  `$DATA_ROOT/new_data` e expor por symlink em `data/new_data/`, como já
  acontece com fluxo e pacotes.
- **G3 — Avaliação normal:** um job GPU que executa `normal --modes
  repo,repo_plus` sobre o teste completo e grava os JSONs e `.npz` em
  `$RESULTS_ROOT/new_data/`.
- **G4 — Avaliação adversarial:** um job array com um ataque por índice, até
  dois concorrentes, para `primary` e, opcionalmente, as sensibilidades
  `sensitivity_first_n` e `sensitivity_paper_0.8`.
- **G5 — Treino e smoke:** tasks para treinar o modelo novo em GPU e para
  rodar o smoke isolado antes da fila longa.
- **G6 — Agregação e operação:** sumarizar os JSONs do novo dataset em
  CSV/Markdown/JSON e documentar o fluxo no runbook.

## 4. Não objetivos

- **Não** alterar `cluster/repo_runner.py`, `cluster/metrics.py` ou o
  comportamento de fluxo/pacotes.
- **Não** alterar `results/flow_based_adversarial_results.json` nem
  `results/packet_based_adversarial_results.json`.
- **Não** mudar o protocolo RePO/RePO+, thresholds, máscaras ou métricas do
  pacote `new_data_eval`.
- **Não** criar um segundo runner de avaliação em `cluster/`; a CLI do pacote
  é a única fonte de scoring e ataque.
- **Não** resolver a citação oficial do dataset nesta entrega.
- **Não** commitar CSVs extraídos ou arquivos `.npz` de scores/checkpoints.

## 5. Decisões

### D1 — Estratégia de integração

Merge com commit de merge de `feat/new-dataset-eval` em
`feat/apuana-cluster`. O conflito de `docs/PROJECT_TASKS.md` é resolvido
mantendo as duas seções (fluxo/pacotes e novo dataset). Os commits de
spec/plan da branch do novo dataset entram junto; não há cherry-pick parcial.

A branch do novo dataset também carrega mudanças herdadas de
`origin/new-dataset` que não pertencem a esta entrega. Depois do merge, o
commit de integração restaura a versão da Apuana para:

- `results/flow_based_adversarial_results.json`;
- `results/packet_based_adversarial_results.json`;
- `code/Flow_based_RePO_Adversarial.ipynb`;
- `code/Packet_based_RePO_Adversarial.ipynb`;
- `code/Colab_Setup.ipynb`;
- renames cosméticos de docs (`COLAB_BROWSER.md`,
  `GPU_TENSORFLOW_COMPATIBILITY.md`) que quebrariam links existentes.

A auditoria do merge compara `git diff` contra `feat/apuana-cluster` e só
mantém mudanças ligadas ao novo dataset, `README`, `docs/PROJECT_TASKS.md`,
`.gitignore`, `requirements.txt` e os arquivos novos (`code/new_data_eval/`,
`code/new_data_*`, `code/train_new_data_model.py`, `data/new_data/`,
`models/new_data_model/`, `extract_new_data.sh`, specs/plans).

### D2 — Layout dos dados no cluster

```text
$DATA_ROOT/new_data/new_train.csv
$DATA_ROOT/new_data/new_test.csv

data/new_data/new_train.csv      -> symlink
data/new_data/new_test.csv       -> symlink
data/new_data/x_train_meta/      -> versionado no repositório
data/new_data/New-dataset-split-clean.zip -> ponteiro LFS versionado
```

O ZIP é puxado com `git lfs pull` e descompactado em `$DATA_ROOT/new_data`.
`extract_new_data.sh` ganha `OUTPUT_DIR` e `ZIP_PATH` configuráveis, mantendo
os defaults atuais para uso local.

Se `git-lfs` não existir no nó de login, o setup usa um fallback que baixa o
objeto LFS pela API do GitHub, validando `oid` e `size` do ponteiro. O fallback
é um helper Python separado, testável sem rede. Ele só é acionado quando o
`git lfs pull` deixa o ZIP como ponteiro; se a API também estiver bloqueada, o
erro explica como copiar o ZIP manualmente para `$DATA_ROOT/new_data`.

O setup trata o novo dataset como etapa própria e controlável:

- `--skip-new-data` pula a extração e o preflight do novo dataset, mantendo o
  fluxo atual de fluxo/pacotes utilizável;
- `--force-new-data` reextrai mesmo que os CSVs já existam;
- por padrão a extração roda, mas uma falha aponta explicitamente para
  `--skip-new-data` se o operador só precisa de fluxo/pacotes.

### D3 — Estrutura dos jobs

- **Normal:** um único job GPU; pontua todas as linhas válidas e deriva RePO e
  RePO+ dos mesmos erros por máscara.
- **Adversarial:** um job array com um ataque por índice, `CONCURRENCY=2`,
  `--checkpoint-every 50`. Cada job executa `--attack <label>`; não usamos
  `--attack all` no cluster para não serializar os nove ataques.
- **Smoke:** job curto e isolado em `$RESULTS_ROOT/new_data/smoke`, rodando
  `validate` + normal truncado + um ataque.
- **Treino:** job GPU separado, com saída em
  `models/retrained/new_data_model`.
- **Export:** task de login, sem GPU.

### D4 — Namespace de comandos

Os comandos existentes permanecem intactos. O novo dataset entra como
subcomandos de `new-data`:

```bash
bash cluster/submit.sh new-data normal
bash cluster/submit.sh new-data adversarial all primary
bash cluster/submit.sh new-data adversarial ddos sensitivity_paper_0.8
bash cluster/submit.sh new-data smoke ddos
bash cluster/submit.sh new-data train
bash cluster/submit.sh new-data export
```

O nome do job SLURM diferencia o dataset (`repo-new-data-*`), e o array usa o
mesmo padrão de lista em `$RESULTS_ROOT/new_data/attack-lists/`.

### D5 — Modelo

- `MODEL_SOURCE=pretrained` (default) usa `models/new_data_model/`.
- `MODEL_SOURCE=retrained` usa `models/retrained/new_data_model/`.
- `MODEL_DIR` continua sendo o override explícito.
- A task `new-data train` recusa gravar em `models/new_data_model/` para não
  sobrescrever o modelo dos autores, do mesmo jeito que fluxo/pacotes. O
  script `code/train_new_data_model.py` continua com o CLI atual; a proteção
  fica no wrapper do cluster para não alterar o comportamento já testado do
  pacote. O runbook avisa que rodar o script localmente com o `--output-dir`
  default sobrescreve o modelo commitado.
- Metadados de normalização vêm de `data/new_data/x_train_meta/`. Se o modelo
  retreinado for usado, o `training_manifest.json` registra o hash do CSV e a
  seed; a comparação de normalização continua sendo responsabilidade do
  operador (o script de treino não grava `x_train_meta`).

### D6 — Diretório de resultados

Tudo do novo dataset fica sob `$RESULTS_ROOT/new_data/`:

```text
$RESULTS_ROOT/new_data/normal/repo.json
$RESULTS_ROOT/new_data/normal/repo_plus.json
$RESULTS_ROOT/new_data/adversarial/primary/<ataque>.json
$RESULTS_ROOT/new_data/adversarial/sensitivity_first_n/<ataque>.json
$RESULTS_ROOT/new_data/adversarial/sensitivity_paper_0.8/<ataque>.json
$RESULTS_ROOT/new_data/scores/*.npz
$RESULTS_ROOT/new_data/smoke/...
$RESULTS_ROOT/new_data/attack-lists/*.txt
```

O export grava `results/cluster/new_data/` por padrão (CSV/MD/JSON pequenos),
com `OUT_DIR` para sobrescrever. Nada é escrito em `results/new_data/` do
repositório durante jobs.

### D7 — Pré-condição do adversarial

O job adversarial canônico exige que
`$RESULTS_ROOT/new_data/normal/repo_plus.json` exista e seja proveniente do
mesmo modelo, dados, metadados, seed, sampling e scoring. Se não existir, o
job falha com instrução para rodar `new-data normal` primeiro. O fallback de
recalcular threshold dentro da CLI continua disponível para uso local, mas não
é o caminho operacional do cluster: recalcular em nove jobs seria caro e
silenciaria um normal ausente.

Variantes e combinações obrigatórias:

| Variante | `--sample-mode` | `--attack-threshold-policy` |
| --- | --- | --- |
| `primary` | `random` | `calibrated` |
| `sensitivity_first_n` | `first` | `calibrated` |
| `sensitivity_paper_0.8` | `random` | `paper_0.8` |

### D8 — Recursos e tempos

Defaults propostos, alinhados ao `submit.sh` atual:

- `PARTITION=short-simple`, `CPUS_PER_TASK=8`, `MEM=32G`, `GRES=gpu:1`;
- `CONCURRENCY=2`;
- `WALLTIME_NEW_DATA_NORMAL=02:00:00`;
- `WALLTIME_NEW_DATA_ADV=1-23:00:00`;
- `WALLTIME_NEW_DATA_TRAIN=1-00:00:00`;
- smoke: `00:30:00`.

O normal novo tem ~3,59 milhões de linhas (mais que os ~2,3 milhões de fluxo),
então 2 h dão margem folgada em A100/L40S: o normal de fluxo medido no Apuana
levou ~10 min para 2,3 M. O adversarial usa até 5.000 amostras por ataque, mas
classes pequenas (`ddos`, `dos`, `mitm`) usam todas as disponíveis. O runbook
recomenda um piloto (`MAX_SAMPLES=200`, `N_MASKS=5`) para medir a vazão antes
da fila longa; os walltimes são defaults, não medições do novo dataset.

### D9 — Agregação

Novo `cluster/aggregate_new_data.py`, independente do agregador de
fluxo/pacotes:

- lê `normal/repo.json`, `normal/repo_plus.json` e
  `adversarial/<variant>/*.json`;
- ignora `smoke/`;
- grava `summary.csv`, `summary.md` e `summary.json`;
- lista resultados ausentes em vez de falhar;
- não depende de TensorFlow.

### D10 — Documentação

- `cluster/README.md`: seção do novo dataset, comandos, estimativas e
  limitações.
- `docs/APUANA_RUNBOOK.md`: passos lineares de setup, smoke, normal,
  adversarial, treino e export.
- `docs/NEW_DATASET.md`: atualizar a nota de escopo (“não usa cluster”) e
  apontar para os comandos do Apuana.
- `cluster/config.env.example`: variáveis novas com defaults comentados.

## 6. Contrato de ambiente dos jobs

`cluster/slurm/job.sh` passa a aceitar:

| Variável | Uso |
| --- | --- |
| `TASK` | `new_data_normal`, `new_data_adversarial`, `new_data_smoke`, `new_data_train` |
| `DATASET` | `new_data` (explícito, para logs) |
| `ATTACK` | label de um ataque (job único) |
| `ATTACK_LIST_FILE` | lista para job array |
| `VARIANT` | `primary`, `sensitivity_first_n`, `sensitivity_paper_0.8` |
| `SAMPLE_MODE` | `random` ou `first` |
| `THRESHOLD_POLICY` | `calibrated` ou `paper_0.8` |
| `MAX_SAMPLES` | amostras por ataque (default 5000) |
| `N_MASKS` | máscaras (default 100; smoke usa 5) |
| `BATCH_SIZE` | linhas mascaradas por forward pass |
| `SCORE_REPEATS` | repetições independentes (default 1) |
| `PAIRED_SCORING` | `0`/`1` |
| `CHECKPOINT_EVERY` | default 50 |
| `NEW_DATA_RESULTS` | default `$RESULTS_ROOT/new_data` |
| `EPOCHS`, `TRAIN_BATCH_SIZE` | treino |
| `FORCE` | descarta resultado/checkpoint/scores |

O job exporta `PYTHONPATH=$REPO_ROOT/code` antes de chamar
`python -m new_data_eval`.

## 7. Riscos e falhas

| Risco | Mitigação |
| --- | --- |
| `git-lfs` ausente no login | fallback pela API do GitHub, validando oid/size |
| ZIP ainda é ponteiro | `extract_new_data.sh` falha com mensagem explícita |
| CSVs ausentes em `data/new_data` | `require_new_data` no setup e no job |
| Normal ainda não rodou | job adversarial falha antes de carregar o modelo |
| Normal truncado/smoke | proveniência da CLI rejeita; só `smoke` aceita |
| Checkpoint de outra config | hash estável recusa; `FORCE=1` descarta |
| OOM no adversarial | `BATCH_SIZE` pode mudar e retomar (fora do hash estável) |
| Dois jobs concorrentes no mesmo ataque | um job por índice; `CONCURRENCY=2` entre ataques |
| Resultados no `$SCRATCH` expiram | export + arquivamento no `$ARCHIVE` |
| Citação do dataset pendente | documentada como pendência, não inventada |

## 8. Critérios de aceitação

1. O merge de `feat/new-dataset-eval` em `feat/apuana-cluster` termina sem
   conflitos não resolvidos.
2. `PYTHONPATH=code python -m pytest code/new_data_eval/tests -q` passa no
   clone integrado.
3. `bash -n` passa em todos os scripts alterados.
4. `bash cluster/setup_env.sh` (com os dados já extraídos) cria os symlinks e o
   preflight do novo dataset sem tocar nos dados de fluxo/pacotes.
5. `bash cluster/submit.sh new-data smoke ddos` imprime o comando interativo e,
   com `SMOKE=1`, submete um job de 30 min.
6. `bash cluster/submit.sh new-data normal` submete um job único e grava
   `normal/repo.json` e `normal/repo_plus.json`.
7. `bash cluster/submit.sh new-data adversarial all primary` submete um job
   array `0-8%2` com os nove ataques e grava um JSON por ataque em
   `adversarial/primary/`.
8. As variantes de sensibilidade gravam em diretórios próprios e não
   sobrescrevem `primary`.
9. `bash cluster/submit.sh new-data export` produz `summary.csv`,
   `summary.md` e `summary.json`, listando resultados ausentes.
10. Nenhum arquivo de `results/flow_based_adversarial_results.json`,
    `results/packet_based_adversarial_results.json` ou `cluster/repo_runner.py`
    é alterado.
11. Nenhum CSV extraído ou `.npz` entra no Git.

## 9. Pendências

- Citação, licença e justificativa formal do dataset permanecem pendentes.
- A execução completa de 9 ataques depende de GPU e do tempo de fila; o design
  entrega o tooling, não os números finais.
- O modelo retreinado não grava `x_train_meta`; se o operador usar
  `MODEL_SOURCE=retrained`, precisa garantir que os metadados correspondem ao
  CSV de treino.
- O `manifest.json` de topo citado na arquitetura do spec do novo dataset não
  faz parte desta integração; a agregação cobre a consolidação.
