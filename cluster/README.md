# Reprodução no cluster Apuana

Este diretório contém o tooling para executar a replicação do RePO no cluster
Apuana (CIn/UFPE), fora do Colab e sem depender de notebooks abertos.

O passo a passo linear, do login até a coleta, está em
[`../docs/APUANA_RUNBOOK.md`](../docs/APUANA_RUNBOOK.md). Este arquivo é a
referência detalhada de cada componente e decisão.

O desenho segue o que já foi validado no projeto `ml_PIDS`:

- ambiente Python em `$SCRATCH`, preparado uma vez no nó de login;
- datasets extraídos para `$SCRATCH` e expostos em `data/` por symlinks;
- um job SLURM por ataque adversarial;
- checkpoint e retomada para jobs de ~23 h;
- resultados em JSON por execução, agregados em CSV/Markdown no final.

## 1. O que é executado

| Etapa | O que faz | Onde roda |
|---|---|---|
| `setup_env.sh` | cria venv, instala TensorFlow, extrai dados, valida modelos | nó de login |
| `train.py` | treina o modelo de fluxo ou de pacotes e exporta SavedModel | job GPU |
| `repo_runner.py normal` | avalia RePO e RePO+ com FPR alvo de 0,01 | job GPU |
| `repo_runner.py adversarial` | executa um ataque com FPR de referência 0,10 | job GPU |
| `aggregate_results.py` | consolida JSONs em CSV/Markdown | login |
| `lfs_fetch.py` | baixa o ZIP do novo dataset sem `git-lfs` | login |
| `aggregate_new_data.py` | consolida os resultados do novo dataset | login |
| `submit.sh` | submete os jobs e imprime os IDs | login |
| `submit.sh new-data ...` | avalia o novo dataset (`data/new_data`) | login |
| `submit.sh mask-exp ...` | roda uma etapa do estudo de máscaras | login |
| `run_mask_study.sh` | submete a cadeia completa do estudo de máscaras com `afterok` | login |

O runner preserva as fórmulas dos notebooks de referência:

- fluxo: 77 features, 100 máscaras, 5 grupos de 20, mínimo por grupo e soma;
- pacotes: janelas de 20 pacotes, 29 features, mesma agregação;
- RePO: média dos erros das 100 máscaras;
- RePO+: mínimo por grupo e soma dos mínimos.

## 2. Pré-requisitos

- VPN do CIn ativa.
- Acesso SSH a `slurm-client1.cin.ufpe.br` (ou `slurm-client2`).
- Chave SSH cadastrada no GitHub para clonar o repositório no cluster.
- Conta com acesso ao cluster e cota no `$SCRATCH`/`$HOME`.

Não rode experimentos nos nós de login. Use-os apenas para `setup`,
submissão, agregação e transferência de arquivos.

## 3. Armazenamento do Apuana

| Área | Uso | Quota | Limpeza |
|---|---|---|---|
| `$HOME` | código, ambiente, resultados pequenos | 100 GB / 1000K arquivos | backup diário |
| `$SCRATCH` | dados extraídos, venv, resultados | sem quota | 90 dias sem uso |
| `$SLURM_TMPDIR` | disco local do job | 4 TB | fim do job |
| `/network/datasets/` | datasets brutos (leitura) | — | — |
| `$ARCHIVE` | resultados de longo prazo | 500 GB | não |

O dataset extraído ocupa cerca de **13 GB** (~1,5 GB de fluxos + ~12 GB de
pacotes). O venv com CUDA ocupa mais alguns GB. Por isso o setup usa
`$SCRATCH` por padrão.

## 4. Clonar e preparar

No nó de login:

```bash
cd "$HOME/Projects" 2>/dev/null || mkdir -p "$HOME/Projects" && cd "$HOME/Projects"
git clone git@github.com:marquesgbr/replicating_RePO.git
cd replicating_RePO
git switch feat/apuana-cluster

export SCRATCH="${SCRATCH:-$HOME}"
bash cluster/setup_env.sh
```

Se o `module` Python não for encontrado automaticamente:

```bash
module avail 2>&1 | grep -i python
export PYTHON_MODULE="Python/3.10"   # use o nome exato listado
bash cluster/setup_env.sh
```

O setup:

1. cria `$SCRATCH/replicating-repo-venv`;
2. instala `cluster/requirements-cluster.txt` (TensorFlow 2.21, Keras 3.15.1);
3. extrai os CSVs em `$SCRATCH/replicating-repo-data/flow_based`;
4. extrai os `.npy` em `$SCRATCH/replicating-repo-data/packet_based`;
5. cria symlinks em `data/` para os arquivos extraídos;
6. grava `pip freeze` em `$SCRATCH/replicating-repo-results/env-lock.txt`.

Para conferir:

```bash
source "$SCRATCH/replicating-repo-venv/bin/activate"
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
ls -lh data/flow_based/Tuesday-WH-generate-labeled.csv
ls -d data/packet_based/{monday,tuesday,wednesday,thursday,friday}
```

## 5. Smoke test (obrigatório antes da fila longa)

A partição `debug` é interativa e tem limite de 30 minutos:

```bash
srun --partition=debug --gres=gpu:1 -c 8 --mem=32G --time=00:30:00 --pty bash
source "$SCRATCH/replicating-repo-venv/bin/activate"
python cluster/repo_runner.py smoke --model flow \
  --results-root "$SCRATCH/replicating-repo-results/smoke"
python cluster/repo_runner.py smoke --model packet \
  --results-root "$SCRATCH/replicating-repo-results/smoke"
```

O smoke valida, em poucos minutos:

- import do TensorFlow e presença de GPU;
- leitura dos CSVs/`.npy`;
- carregamento do SavedModel pelo `load_model_compat`;
- pontuação normal e um passo de ataque;
- escrita dos JSONs em `$RESULTS_ROOT`.

Se falhar por falta de GPU, confira `nvidia-smi` e a partição. Se falhar por
dados/modelos, volte ao setup.

## 6. Piloto

Comece pelo cenário mais curto e pelo teste normal, comparando com os números
já conhecidos do repositório:

```bash
bash cluster/submit.sh normal flow
bash cluster/submit.sh normal packet
bash cluster/submit.sh adversarial flow "DoS Slowhttptest"
```

Acompanhe:

```bash
squeue -u "$USER"
tail -f "$SCRATCH/replicating-repo-results/logs/"*.out
```

O ataque de fluxo `DoS Slowhttptest` levou ~186 min na máquina usada antes.
O ataque de pacotes `FTP-Patator` levou ~23 h; use-o como piloto apenas depois
de validar o fluxo e o smoke de pacotes.

## 7. Execução completa

```bash
# 11 ataques de fluxo, um job por ataque, 2 concorrentes
bash cluster/submit.sh adversarial flow all

# 11 ataques de pacotes (Web-Attack fica fora, como no artigo)
bash cluster/submit.sh adversarial packet all
```

O script cria uma lista em
`$RESULTS_ROOT/attack-lists/<modelo>-<timestamp>.txt` e submete um job array
com `--array=0-N%CONCURRENCY`. Cada índice executa um ataque diferente.

### Loop compilado no ataque de pacotes (`PACKET_GPU_LOOP=1`)

Os notebooks originais fazem 300 iterações de otimização por pacote em Python,
sincronizando CPU↔GPU a cada passo. A branch `main` adotou um `tf.while_loop`
compilado (`compiled_delay_loop` / `compiled_inject_loop`); o runner reproduz a
mesma otimização atrás da flag `--packet-gpu-loop`:

```bash
FORCE=1 PACKET_GPU_LOOP=1 bash cluster/submit.sh adversarial packet all
```

Validação A/B em 200 pacotes do FTP-Patator, mesma GPU e mesma seed:

| Caso | Tempo do ataque | victim_tpr | fooled |
|---|---|---|---|
| sem otimização | 245,1 s | 0,6350 | 3 |
| `PACKET_GPU_LOOP=1` | **94,3 s** | 0,6350 | 3 |

Ganho de **2,6×** com resultados idênticos. A flag entra no `config_hash`, então
checkpoints antigos não são reaproveitados por engano (use `FORCE=1` para
descartá-los).

Estimativas na GPU usada antes (RTX 2060 sem o loop compilado):

| Modelo | Ataques | Tempo por ataque | 2 concorrentes |
|---|---|---|---|
| fluxo | 11 | ~3 h | ~17 h |
| pacotes (loop Python) | 11 | ~23 h | ~6 dias |
| pacotes (loop compilado) | 11 | ~9 h | ~2 dias |

Em A100/L40S o tempo deve cair; o gargalo dos pacotes é o número de iterações
de otimização por pacote, não só a GPU.

## 8. Partições e recursos

| Partição | Interativa | Recursos | Tempo máx. |
|---|---|---|---|
| `install` | sim | meio nó | 30 min |
| `debug` | sim | nó completo | 30 min |
| `short-simple` | não | meio nó (≤2 GPU) | 2 dias |
| `short-complex` | não | nó completo (≤4 GPU) | 2 dias |
| `long-simple` | não | meio nó (≤2 GPU) | 7 dias |
| `long-complex` | não | nó completo (≤4 GPU) | 7 dias |
| `emergency` | não | nó completo | 1 dia |

### Limites observados na prática (nov/2026)

| Limite | Valor observado | Consequência |
|---|---|---|
| GPUs simultâneas por usuário | **2** | `CONCURRENCY=2` no máximo |
| Memória simultânea por usuário | ~32 GB no total | use `MEM=16G` por job |
| `--nodelist` com vários hosts | pede 1 nó por host | use `--exclude` para evitar nós ruins |
| Nós com defeito | `cluster-node4`, `cluster-node10` | `EXCLUDE="cluster-node4,cluster-node10"` |
| Partição `debug` | jobs falhavam no launch | smoke em `short-simple` |
| AVX no nó de login | ausente | não importe TensorFlow no login |
| `/opt/miniforge3` | só no login | venv com `/usr/bin/python3.10` |

Defaults do `submit.sh`:

- `PARTITION=short-simple`
- `CPUS_PER_TASK=8`
- `MEM=16G`
- `GRES=gpu:1`
- `CONCURRENCY=2`
- `EXCLUDE="cluster-node4,cluster-node10"`
- `WALLTIME_ADV=1-23:00:00` (47 h; cabe nos 2 dias do `short-simple`)
- `WALLTIME_NORMAL=06:00:00`
- `WALLTIME_TRAIN=1-00:00:00`

### Custos medidos (A100 80 GB, nov/2026)

| Etapa | Custo |
|---|---|
| Normal fluxo (2,3 M amostras, RePO+RePO+) | ~10 min |
| Normal pacotes (2,2 M janelas, RePO+RePO+) | ~5 min |
| Adversarial fluxo, FTP-Patator (5.000) | ~14 min |
| Adversarial fluxo, Slowloris/Slowhttptest | ~1,5–2 h |
| Adversarial pacotes (piloto, 81 pacotes) | ~1,0 s/pacote → ~7 h para 25.000 |

Para usar A100, por exemplo:

```bash
NODELIST=cluster-node6 bash cluster/submit.sh adversarial flow all
```

Para deixar o ataque de pacotes em partição de 7 dias:

```bash
PARTITION=long-simple WALLTIME_ADV=6-00:00:00 \
  bash cluster/submit.sh adversarial packet all
```

Valide a submissão sem executar:

```bash
sbatch --test-only --partition=short-simple --gres=gpu:1 \
  -c 8 --mem=32G --time=1-23:00:00 cluster/slurm/job.sh
```

## 9. Retomada e checkpoint

Os jobs adversariais salvam checkpoints:

- fluxo: a cada `CHECKPOINT_EVERY` amostras (50 por padrão);
- pacotes: a cada `CHECKPOINT_EVERY` pacotes (50 por padrão).

Se o job for interrompido por preempção, `scancel` ou `SIGTERM`, o runner salva
o checkpoint antes de sair. Para retomar, reenvie o mesmo ataque:

```bash
bash cluster/submit.sh adversarial packet "FTP-Patator"
```

O runner detecta o checkpoint, valida o `config_hash` e continua do ponto
salvo. Se a configuração mudou (modelo, seed, threshold, `max_samples`), ele
recusa continuar e pede `FORCE=1`:

```bash
FORCE=1 bash cluster/submit.sh adversarial packet "FTP-Patator"
```

Um resultado já concluído é ignorado em novas submissões (idempotência).

## 10. Treino

O treino usa os mesmos hiperparâmetros do artigo:

- fluxo: Monday, 5 épocas, batch 256, Adam 1e-3, máscara 0,75;
- pacotes: Monday, 30.000 batches (10.000 por LR), batch 512,
  Adam 1e-3/1e-4/1e-5, máscara 0,75.

```bash
bash cluster/submit.sh train flow
bash cluster/submit.sh train packet
```

Os modelos são gravados em `models/retrained/flw_model` e
`models/retrained/pkt_model`; os modelos dos autores em `models/` nunca são
sobrescritos. O min/max de normalização fica em `<modelo>/x_train_meta/`.

Para avaliar os modelos retreinados:

```bash
MODEL_SOURCE=retrained bash cluster/submit.sh normal flow
MODEL_SOURCE=retrained bash cluster/submit.sh adversarial flow all
```

O treino é opcional para a reprodução dos números do artigo (os modelos dos
autores já estão no repositório), mas é útil para comparar o pipeline completo.
O treino não tem checkpoint: se o job for interrompido, reenvie-o.

## 11. Agregação e coleta

```bash
bash cluster/submit.sh export
```

Gera em `results/cluster/` (no repositório):

- `summary.csv` — uma linha por resultado;
- `summary.md` — tabelas normal/adversarial e lista de resultados ausentes;
- `summary.json` — payload completo.

Para arquivar os resultados brutos antes da limpeza de 90 dias do `$SCRATCH`:

```bash
tar cJf "${ARCHIVE:-$HOME}/replicating-repo-results-$(date -u +%Y%m%d).tar.xz" \
  -C "$SCRATCH/replicating-repo-results" .

# da máquina local:
rsync --bwlimit=1000 -azP \
  <login>@slurm-client1.cin.ufpe.br:"$SCRATCH/replicating-repo-results" \
  ~/Projects/replicating_RePO/
```

## 12. Métricas e thresholds

O artigo usa dois pontos de operação:

- **normal:** FPR = 0,01; avalia RePO e RePO+;
- **adversarial:** FPR = 0,10; avalia RePO+.

Os thresholds de referência do artigo estão em `cluster/metrics.py`:

- fluxo: `real_threshold = 0.016235964372754097` e
  `attack_threshold = 0.8 * real_threshold`;
- pacotes: `threshold = 0.08106673508882523`.

Os campos dos resultados têm semântica explícita:

| Campo | Significado |
|---|---|
| `normal_tpr` | TPR nas amostras originais, no threshold de referência |
| `victim_tpr` | TPR depois do ataque (fluxo: re-pontuação; pacotes: `cons_as_mal/n`) |
| `attack_success_rate` | `1 - victim_tpr` |
| `fooled_rate` | fração de amostras com modificação adversarial bem-sucedida |
| `counts` | `cons_as_mal`, `cons_as_ben`, `fooled`, `n` |

O runner normal salva também `scores_<modo>.npz`, permitindo recalcular
thresholds sem repetir a avaliação.

## 13. Desvios conscientes em relação aos notebooks

1. **Fluxo normal normalizado.** O notebook normal de fluxo chama o modelo com
   os dados sem normalizar (`get_test_set` tem a normalização comentada). Isso
   não corresponde ao treino, que normaliza com min/max. O runner normaliza
   sempre. O notebook original fica intacto para referência.
2. **Métrica de pacotes corrigida.** O notebook adversarial de pacotes salvava
   `attacker_tpr` como o TPR normal, duplicando `normal_tpr`. O runner grava
   `normal_tpr`, `victim_tpr`, `attack_success_rate` e `counts` separadamente.
3. **Seed fixa.** Os notebooks não fixam seed; o runner usa `SEED=2019` por
   padrão para permitir retomada e repetição. As máscaras continuam
   estocásticas, mas reproduzíveis.
4. **RePO no fluxo.** O notebook normal de fluxo só implementa RePO+; o runner
   calcula RePO (média das 100 máscaras) e RePO+ (mínimo por grupo e soma).
5. **Último lote de pacotes.** O notebook normal de pacotes descartava o último
   lote parcial de 50 janelas; o runner pontua todas as janelas. A diferença é
   de no máximo 49 janelas em ~2,2 milhões.
6. **Máscaras por janela de pacotes.** O notebook de pacotes dependia de um
   broadcasting implícito entre o lote e as 100 máscaras; o runner aplica
   explicitamente 100 máscaras por janela, que é o procedimento descrito no
   artigo.

## 14. Diagnóstico

- **`Illegal instruction` ao importar TensorFlow no login**: o nó de login do
  Apuana não tem AVX. Isso é esperado; valide TensorFlow dentro de um job.
- **`python: No such file or directory` no job**: o venv foi criado com o
  Python do miniforge, que só existe nos nós de login. Recrie com
  `/usr/bin/python3.10` (`bash cluster/setup_env.sh --force-venv`).
- **`module` não encontrado**: rode `source /etc/profile` ou use um shell de
  login; o setup avisa e tenta o `python3` do PATH.
- **TensorFlow não vê GPU**: confira `nvidia-smi`, a partição e se o job pediu
  `--gres=gpu:1`. O setup instala CUDA via `tensorflow[and-cuda]`.
- **Job preso em `PD`**: veja `squeue -j <id> -o %r`; os limites de QoS são
  16 CPUs/64 GB/2 GPUs em `*-simple` e 48 CPUs/500 GB/4 GPUs em `*-complex`.
- **Pouco espaço**: `df -h "$SCRATCH"`; os resultados podem ser movidos para
  `$ARCHIVE` ou para a máquina local.
- **Retomada recusada**: o `config_hash` mudou; use `FORCE=1` somente se
  realmente quiser descartar o checkpoint.
- **Logs**: `$RESULTS_ROOT/logs/<job>.out`, `.err` e `$RESULTS_ROOT/logs/*.log`.

## 15. Ordem recomendada

1. `setup_env.sh` no login.
2. smoke de fluxo e pacotes em `debug`.
3. `normal flow` e `normal packet` (piloto).
4. `adversarial flow "DoS Slowhttptest"` (piloto).
5. `adversarial packet "FTP-Patator"` (piloto longo).
6. `adversarial flow all`.
7. `adversarial packet all`.
8. `export` e arquivamento.
9. Atualizar `docs/PROJECT_TASKS.md` e o relatório com os números.

Para o novo dataset, o fluxo equivalente é:

1. `setup_env.sh` (extrai `data/new_data`; use `--skip-new-data` para pular).
2. `SMOKE=1 bash cluster/submit.sh new-data smoke ddos`.
3. `bash cluster/submit.sh new-data normal`.
4. `bash cluster/submit.sh new-data adversarial all primary`.
5. Sensibilidades opcionais (`sensitivity_first_n`, `sensitivity_paper_0.8`).
6. `bash cluster/submit.sh new-data export`.

## 16. Novo dataset (`data/new_data`)

O pacote `code/new_data_eval/` roda no Apuana pelo mesmo esquema dos outros
datasets, mas com tasks próprias. Ele **não** altera o runner, os resultados ou
os notebooks de fluxo/pacotes.

### 16.1 Dados e modelo

O setup extrai o ZIP LFS para `$DATA_ROOT/new_data` e cria symlinks:

```text
$DATA_ROOT/new_data/new_train.csv
$DATA_ROOT/new_data/new_test.csv

data/new_data/new_train.csv  -> symlink
data/new_data/new_test.csv   -> symlink
```

Metadados de normalização (`data/new_data/x_train_meta/`) e o SavedModel
(`models/new_data_model/`) são versionados no repositório. Se `git-lfs` não
existir no login, `extract_new_data.sh` baixa o objeto pela API do GitHub,
verifica tamanho/SHA-256 e grava o ZIP real em `$DATA_ROOT/new_data/` — o
ponteiro LFS versionado não é sobrescrito. Se a rede bloquear, copie o ZIP
manualmente para `$DATA_ROOT/new_data/` ou use `--skip-new-data`.

Flags úteis:

```bash
bash cluster/setup_env.sh                 # extrai o novo dataset
bash cluster/setup_env.sh --skip-new-data # só fluxo/pacotes
bash cluster/setup_env.sh --force-new-data
```

### 16.2 Comandos

```bash
# Smoke isolado em $RESULTS_ROOT/new_data/smoke (30 min)
SMOKE=1 bash cluster/submit.sh new-data smoke ddos

# Avaliação normal completa: um job, RePO e RePO+
bash cluster/submit.sh new-data normal

# Adversarial: um job array, um ataque por índice, 2 concorrentes
bash cluster/submit.sh new-data adversarial all primary

# Sensibilidades (diretórios próprios; não sobrescrevem primary)
bash cluster/submit.sh new-data adversarial all sensitivity_first_n
bash cluster/submit.sh new-data adversarial all sensitivity_paper_0.8

# Treino opcional em models/retrained/new_data_model
bash cluster/submit.sh new-data train

# Resumo CSV/Markdown/JSON em results/cluster/new_data
bash cluster/submit.sh new-data export
```

O adversarial canônico exige que o normal já tenha terminado:

```text
$RESULTS_ROOT/new_data/normal/repo_plus.json
```

Se o arquivo não existir, o job falha antes de carregar o modelo. Rode o normal
primeiro; não deixe nove jobs recalcularem o threshold.

### 16.3 Variantes

| Variante | Amostragem | Threshold |
| --- | --- | --- |
| `primary` | aleatória (seed 2019) | calibrado no FPR 10% |
| `sensitivity_first_n` | primeiras N | calibrado no FPR 10% |
| `sensitivity_paper_0.8` | aleatória | `0,8 ×` threshold real |
| `smoke` | aleatória | calibrado no FPR 10%; inspeção, não resultado |

### 16.4 Piloto e estimativas

Antes do normal, valide o ambiente com `new-data smoke`. Depois que o normal
terminar, meça a vazão adversarial com um piloto:

```bash
MAX_SAMPLES=200 BATCH_SIZE=256 \
  bash cluster/submit.sh new-data adversarial ddos smoke
```

Use `smoke` no piloto: `primary` guarda o resultado canônico e um piloto com
`MAX_SAMPLES` menor bloquearia a rodada completa por causa do checkpoint. Não
troque `N_MASKS`, `SCORE_REPEATS` ou `PAIRED_SCORING` no adversarial canônico:
a proveniência exige os mesmos valores usados no normal. Para inspecionar com
5 máscaras, use `new-data smoke`.

O normal percorre ~3,59 M linhas com 100 máscaras; no A100 usado antes, o normal
de fluxo (2,3 M) levou ~10 min, então 2 h dão margem. O adversarial usa até
5.000 amostras por ataque; `ddos` (202), `dos` (145) e `mitm` (517) usam todas
as disponíveis.

### 16.5 Resultados e retomada

```text
$RESULTS_ROOT/new_data/normal/repo.json
$RESULTS_ROOT/new_data/normal/repo_plus.json
$RESULTS_ROOT/new_data/adversarial/<variante>/<ataque>.json
$RESULTS_ROOT/new_data/scores/*.npz
$RESULTS_ROOT/new_data/smoke/...
```

Checkpoint a cada 50 amostras. Retome reenviando o mesmo comando; se a
configuração mudou, use `FORCE=1`. `BATCH_SIZE` pode mudar numa retomada após
OOM porque não entra no hash do checkpoint.

### 16.6 Rodada autônoma (dependências SLURM)

`DEPENDENCY` é repassado ao `sbatch` em qualquer submissão. Para deixar a
cadeia rodando sem depender da sua máquina:

```bash
# 1. normal (guarde o ID)
NORMAL_JOB=$(bash cluster/submit.sh new-data normal | awk 'NR==1{print $NF}')

# 2. adversarial só começa se o normal terminar com sucesso
ARRAY_JOB=$(DEPENDENCY="afterok:${NORMAL_JOB}" \
  bash cluster/submit.sh new-data adversarial all primary | awk 'NR==1{print $NF}')

# 3. exportação só roda se todos os ataques do array terminarem bem
DEPENDENCY="afterok:${ARRAY_JOB}" \
  bash cluster/submit.sh new-data export-job
```

Para rodar também as sensibilidades, capture cada array e encadeie o export
com `afterok:id1:id2`. O `export-job` roda sem GPU e grava em
`$RESULTS_ROOT/new_data/export/` (`summary.csv`, `summary.md`,
`summary.json`). Se o normal falhar, o adversarial não começa; se um ataque
falhar, o export não roda. Os jobs já submetidos continuam mesmo se você
desligar o computador local; os checkpoints cobrem preempção/requeue.

## 17. Eficiência do sistema de máscaras (`mask-exp`)

O namespace `mask-exp` avalia as três direções do estudo de máscaras com o
mesmo harness local de `code/mask_experiments/`.  Todos os resultados vão para
`$RESULTS_ROOT/mask_efficiency/`; os jobs adversariais são arrays com um job
por ataque e o agregador junta os JSONs e gera o Markdown.

### 17.1 Sequência recomendada

A matriz completa (status de cada run, escala, custo e métricas afetadas) está
em [`../docs/MASK_EFFICIENCY_RUNS.md`](../docs/MASK_EFFICIENCY_RUNS.md).

**Cadeia completa em um comando** (no nó de login, dentro do repositório):

```bash
bash cluster/run_mask_study.sh            # produção
bash cluster/run_mask_study.sh --smoke    # valida a cadeia inteira
bash cluster/run_mask_study.sh --dry-run  # só imprime o plano
```

O script submete as etapas com dependências `afterok`, grava
`$RESULTS_ROOT/mask_efficiency/chain_manifest.tsv` e passa `MASK_STEP_NOTE`
para cada job; o log de cada job começa com `[step] <descrição>`.  Se uma
etapa falhar, as seguintes não começam.  Para retomar, reenvie só a etapa que
falhou e depois o `aggregate`.

Etapas e o que cada uma faz:

| Etapa | O que faz | Saída |
|---|---|---|
| `baseline` | congela thresholds/TPR do RePO e RePO+ (Gate G1) | `baseline_flow.json` |
| `variance` | curva TPR normal × `n_masks` × política e custo | `variance_flow.json` |
| `adversarial` | ataque adaptativo nas 9 variantes (array por ataque) | `adversarial/<ataque>.json` |
| `cascade` | cascata + RePO sequencial na mistura 99/1 | `cascade_flow.json` |
| `cascade_attack` | ataque ao estágio barato (array por ataque) | `cascade_attack/<ataque>.json` |
| `distill` | treina o aluno de passe único | `distillation_flow.json` + `.keras` |
| `distill_attack` | ataque white-box no aluno, professor pareado | `distill_attack/<ataque>.json` |
| `packet` | confirmação normal no modelo de pacotes | `packet_normal.json` |
| `aggregate` | merge dos arrays + Markdown consolidado | `MASK_EFFICIENCY_RESULTS.md` |

Para submeter uma etapa isolada (ou depurar), use os comandos manuais:

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

`mask-exp distill-attack` exige que `distillation_student.keras` já exista;
submeta o `distill` primeiro (ou use `DEPENDENCY=afterok:<id>`, que é o que a
cadeia faz). O `cascade-attack` recalcula os próprios thresholds e não depende
do `cascade`.

### 17.2 Métricas adversariais por direção

| Direção | O que é medido | Onde |
|---|---|---|
| A — máscaras | TPR de vítima no FPR casado, `fooled`, `no_change`, máscaras/ataque | `adversarial/<ataque>.json` |
| B — cascata | TPR de vítima da cascata, `bypass` do estágio barato, `fooled` | `cascade_attack/<ataque>.json` |
| C — destilação | TPR de vítima do aluno vs. professor nos **mesmos** exemplos | `distill_attack/<ataque>.json` |

O TPR de vítima é sempre medido com o threshold do próprio detector,
recalibrado no FPR de 10 %; o ataque de referência usa a mesma projeção de
viabilidade e o atacante conhece a política/número de máscaras do detector.

### 17.3 Defaults de produção e smoke

Os defaults do `mask-exp` são de produção (`MASK_MAX_BENIGN=100000`,
`MASK_MAX_PER_ATTACK=5000`, `MASK_PER_ATTACK=5000`, 11 ataques).  Para um
smoke barato de **todas as etapas**:

```bash
bash cluster/run_mask_study.sh --smoke
```

Para um smoke de uma etapa isolada:

```bash
MASK_MAX_BENIGN=2000 MASK_MAX_PER_ATTACK=200 MASK_PER_ATTACK=20 \
  MASK_ATTACK_SET=pilot bash cluster/submit.sh mask-exp adversarial all

MASK_MAX_BENIGN=2000 MASK_MAX_PER_ATTACK=200 MASK_PER_ATTACK=20 \
  bash cluster/submit.sh mask-exp distill-attack all
```

O cache de amostragem fica desligado nos jobs (`MASK_CACHE` vazio) para evitar
corrida entre os jobs do array; cada job reamostra as CSVs (≈13 s).

### 17.4 Saídas

```
$RESULTS_ROOT/mask_efficiency/
  baseline_flow.json
  variance_flow.json
  adversarial/<ataque>.json
  adversarial_all.json          # merge do array
  cascade_flow.json
  cascade_attack/<ataque>.json
  cascade_attack_all.json
  distillation_flow.json
  distillation_student.keras
  distill_attack/<ataque>.json
  distill_attack_all.json
  packet_normal.json
  MASK_EFFICIENCY_RESULTS.md    # relatório consolidado
```

O `mask-exp aggregate` roda em CPU, faz o merge dos arrays e chama
`python -m mask_experiments.aggregate`.  Para copiar o relatório para o
repositório, `cp` do `$RESULTS_ROOT` ou use o `export` do projeto.

